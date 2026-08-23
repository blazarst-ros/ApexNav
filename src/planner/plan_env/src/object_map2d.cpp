/**
 * @file object_map2d.cpp
 * @brief Implementation of semantic object mapping system for autonomous navigation
 *
 * This file contains the complete implementation of the ObjectMap2D class,
 * providing semantic object detection, multi-view fusion, confidence scoring,
 * and 3D point cloud processing capabilities. The system integrates vision-based
 * object detection with occupancy mapping to create robust semantic representations.
 *
 * @author Zager-Zhang
 */

#include <plan_env/object_map2d.h>

#include <algorithm>

namespace apexnav_planner {
ObjectMap2D::ObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
  exposure_rank_pending_ = false;
  // Initialize core mapping components
  this->sdf_map_ = sdf_map;
  int voxel_num = sdf_map_->getVoxelNum();
  object_buffer_ = vector<char>(voxel_num, 0);  // Object occupancy flags per grid cell
  object_indexs_ = vector<int>(voxel_num, -1);  // Object ID mapping per grid cell

  // Initialize point cloud containers
  all_object_clouds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  over_depth_object_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // Load configuration parameters
  min_confidence_ = -1.0;  // Default to accept all detections
  nh.param("object/min_observation_num", min_observation_num_, 2);
  nh.param("object/fusion_type", fusion_type_, 1);
  nh.param("object/use_observation", use_observation_, true);
  nh.param("object/vis_cloud", is_vis_cloud_, false);
  nh.param("object/exposure_heatmap_enabled", exposure_heatmap_enabled_, true);
  nh.param("object/exposure_capacity", exposure_capacity_, 2.0);
  double exposure_hfov_deg;
  nh.param("object/exposure_hfov_deg", exposure_hfov_deg, 79.0);
  exposure_hfov_rad_ = exposure_hfov_deg * M_PI / 180.0;

  // Setup ROS communication
  object_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object/clouds", 10);
  exposure_heatmap_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object/exposure_heatmap", 10);

  // Configure raycasting for spatial queries
  raycaster_.reset(new RayCaster2D);
  resolution_ = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  // Set point cloud processing parameters
  leaf_size_ = 0.1f;  // Voxel grid leaf size for downsampling
}

void ObjectMap2D::setConfidenceThreshold(double val)
{
  min_confidence_ = val;
  ROS_INFO("Set Confidence Threshold = %f", val);
}

void ObjectMap2D::setExposureObservationContext(const std::string& scene_id,
    const std::string& episode_id, uint32_t step_index, const Vector3d& camera_position,
    double camera_yaw)
{
  if ((!exposure_scene_id_.empty() || !exposure_episode_id_.empty()) &&
      (scene_id != exposure_scene_id_ || episode_id != exposure_episode_id_)) {
    clearExposureHeatmap();
  }
  exposure_scene_id_ = scene_id;
  exposure_episode_id_ = episode_id;
  exposure_step_index_ = step_index;
  exposure_camera_position_ = camera_position;
  exposure_camera_yaw_ = camera_yaw;
}

vector<ExposureHeatmapUpdate> ObjectMap2D::consumeExposureUpdates()
{
  vector<ExposureHeatmapUpdate> updates;
  updates.swap(pending_exposure_updates_);
  return updates;
}

void ObjectMap2D::clearExposureHeatmap()
{
  int cleared_clusters = 0;
  for (auto& object : objects_) {
    if (!object.exposure_by_grid_.empty()) {
      object.exposure_by_grid_.clear();
      ++cleared_clusters;
    }
  }
  pending_exposure_updates_.clear();
  exposure_rank_pending_ = false;
  ROS_INFO("[ExposureHeatmap][RESET] scene=%s episode=%s cleared_clusters=%d",
      exposure_scene_id_.c_str(), exposure_episode_id_.c_str(), cleared_clusters);
}

/**
 * @brief Process observation clouds to adjust detection confidence
 *
 * This function handles negative evidence from visual observations where
 * objects were expected but not detected. It computes spatial overlap
 * between observation regions and existing detections to reduce confidence
 * scores, improving the robustness of the semantic mapping system.
 *
 * @param observation_clouds Vector of point clouds representing observed regions
 * @param itm_score Image-text matching score for context weighting
 */
void ObjectMap2D::inputObservationObjectsCloud(
    const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> observation_clouds,
    const double& itm_score)
{
  // Only process observations in fusion mode with observation enabled
  if (fusion_type_ != 1 || !use_observation_)
    return;

  // Process each observation cloud against corresponding objects
  for (int i = 0; i < (int)observation_clouds.size(); i++) {
    auto observation_cloud = observation_clouds[i];
    auto object = objects_[i];

    if (observation_cloud->points.empty())
      continue;

    // Check overlap with each possible object classification
    for (int label = 0; label < 5; ++label) {
      if (object.confidence_scores_[label] < 1e-3)
        continue;  // Skip labels with negligible confidence

      // Setup spatial search for overlap computation
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud(observation_cloud);
      double distance_threshold = leaf_size_ * 1.1;  // Spatial overlap threshold
      int overlap_count = 0;

      // Count overlapping points between object and observation clouds
      for (const auto& point : object.clouds_[label]->points) {
        std::vector<int> point_idx_search;
        std::vector<float> point_squared_distance;
        if (kdtree.nearestKSearch(point, 1, point_idx_search, point_squared_distance) > 0) {
          // Points within threshold are considered overlapping
          if (point_squared_distance[0] <= distance_threshold * distance_threshold) {
            overlap_count++;
          }
        }
      }

      // Skip if no spatial overlap detected
      if (overlap_count == 0)
        continue;

      // Update confidence scores based on negative observation evidence
      auto& merged_object = objects_[i];
      merged_object.observation_cloud_sums_[label] += overlap_count;
      int total_last = merged_object.clouds_[label]->points.size();
      double confidence_last = merged_object.confidence_scores_[label];
      int observation_now = overlap_count;
      double confidence_now = 0.0;  // Negative evidence has zero confidence
      if (label == 0)
        confidence_now = itm_score;  // Use ITM score for primary label
      int total_now = merged_object.clouds_[label]->points.size();

      // Apply confidence fusion algorithm
      merged_object.confidence_scores_[label] = fusionConfidenceScore(total_last, confidence_last,
          observation_now, confidence_now, total_now, merged_object.observation_cloud_sums_[label]);
      printFusionInfo(merged_object, label, "[Observation]");
      // ROS_WARN("[Observation] id = %d label = %d overlap_count = %d object_cloud = %ld",
      //     merged_object.id_, label, overlap_count, object.clouds_[label]->points.size());
    }
  }
}

int ObjectMap2D::searchSingleObjectCluster(const DetectedObject& detected_object)
{
  auto object_cloud = detected_object.cloud;

  // Initialize clustering analysis variables
  int point_num = object_cloud->points.size();
  int obj_idx = -1;
  vector<Eigen::Vector2d> object_point2Ds;
  vector<char> flag_2d(sdf_map_->getVoxelNum(), 0);  // Duplicate point prevention

  // Process each point in the detected object cloud
  for (int i = 0; i < point_num; i++) {
    Eigen::Vector2i idx;
    Eigen::Vector2d pt_w;
    pt_w << object_cloud->points[i].x, object_cloud->points[i].y;
    sdf_map_->posToIndex(pt_w, idx);
    int adr = sdf_map_->toAddress(idx);

    // Skip duplicate grid cells
    if (flag_2d[adr] == 1)
      continue;

    flag_2d[adr] = 1;

    // Validate if point satisfies object characteristics
    if (isSatisfyObject(pt_w)) {
      object_buffer_[adr] = 1;  // Mark cell as containing object
      object_point2Ds.push_back(pt_w);
    }
  }

  // Return early if no valid object points found
  if (object_point2Ds.empty()) {
    return -1;
  }

  // Search for existing object clusters in neighborhood
  for (auto pt_w : object_point2Ds) {
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(pt_w, idx);

    // Get neighboring grid cells within clustering radius
    auto nbrs = allGridsDistance(idx, 0.08);
    nbrs.push_back(idx);

    // Check neighbors for existing object associations
    for (auto nbr : nbrs) {
      int nbr_adr = sdf_map_->toAddress(nbr);
      if (object_indexs_[nbr_adr] != -1) {
        // Found existing object cluster - use first match
        // TODO: Implement multi-object merging for complex scenarios
        obj_idx = object_indexs_[nbr_adr];
        break;
      }
    }
  }

  // Apply voxel grid filtering to reduce point cloud density
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(object_cloud);
  voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  voxel_filter.filter(*detected_object.cloud);

  // Either merge with existing cluster or create new one
  if (obj_idx != -1) {
    mergeCellsIntoObjectCluster(obj_idx, object_point2Ds, detected_object);
  }
  else {
    createNewObjectCluster(object_point2Ds, detected_object);
    obj_idx = object_indexs_[toAdr(object_point2Ds[0])];
  }

  // Update classification and visualization
  updateObjectBestLabel(obj_idx);
  updateExposureHeatmap(obj_idx, detected_object.label, object_point2Ds);
  if (is_vis_cloud_)
    publishObjectClouds();

  // Validate successful clustering
  if (obj_idx == -1) {
    ROS_ERROR("[bug] Why not find the object cluster!?");
    return obj_idx;
  }

  return obj_idx;
}

void ObjectMap2D::updateObjectBestLabel(int obj_idx)
{
  double max_func_score = 0.1;  // Minimum threshold for valid classification
  int best_label = -1;

  // Evaluate each possible classification label
  for (int label = 0; label < (int)objects_[obj_idx].clouds_.size(); label++) {
    auto obs_sum = objects_[obj_idx].observation_cloud_sums_[label];
    auto score = objects_[obj_idx].confidence_scores_[label];
    int func_score = obs_sum * score;  // Combined reliability metric

    if (func_score > max_func_score) {
      max_func_score = func_score;
      best_label = label;
    }
  }
  objects_[obj_idx].best_label_ = best_label;
}

void ObjectMap2D::createNewObjectCluster(
    const std::vector<Eigen::Vector2d>& cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;

  // Initialize new object cluster with unique ID
  ObjectCluster obj;
  obj.id_ = (int)objects_.size();
  obj.max_seen_count_ = 0;
  obj.good_cells_.clear();
  obj.seen_counts_.clear();
  obj.best_label_ = -1;

  // Process spatial cells and establish grid associations
  std::vector<Eigen::Vector2d> real_new_cells;
  for (auto cell : cells) {
    int adr = toAdr(cell);
    object_indexs_[adr] = obj.id_;  // Associate grid cell with object
    obj.visited_[adr] = 1;

    // Track high-confidence observations for label 0
    if (label == 0) {
      obj.seen_counts_[adr] = 1;
      obj.max_seen_count_ = 1;
      obj.good_cells_.push_back(cell);
    }
  }

  // Compute spatial properties of the object cluster
  obj.cells_ = cells;
  obj.average_.setZero();
  obj.box_max2d_ = obj.cells_.front();
  obj.box_min2d_ = obj.cells_.front();

  for (auto cell : obj.cells_) {
    obj.average_ += cell;
    for (int i = 0; i < 2; ++i) {
      obj.box_min2d_[i] = min(obj.box_min2d_[i], cell[i]);
      obj.box_max2d_[i] = max(obj.box_max2d_[i], cell[i]);
    }
  }
  obj.average_ /= double(obj.cells_.size());

  // Initialize point cloud storage for the detected label
  obj.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
  *obj.clouds_[label] = *detected_object.cloud;

  // Compute 3D bounding box from point cloud
  obj.box_max3d_ = Vector3d(obj.clouds_[label]->points[0].x, obj.clouds_[label]->points[0].y,
      obj.clouds_[label]->points[0].z);
  obj.box_min3d_ = Vector3d(obj.clouds_[label]->points[0].x, obj.clouds_[label]->points[0].y,
      obj.clouds_[label]->points[0].z);

  for (auto pt : obj.clouds_[label]->points) {
    Vector3d vec_pt = Vector3d(pt.x, pt.y, pt.z);
    for (int i = 0; i < 3; ++i) {
      obj.box_min3d_[i] = min(obj.box_min3d_[i], vec_pt[i]);
      obj.box_max3d_[i] = max(obj.box_max3d_[i], vec_pt[i]);
    }
  }

  // Initialize confidence tracking for this object
  obj.confidence_scores_[label] = detected_object.score;
  obj.observation_cloud_sums_[label] = detected_object.cloud->points.size();
  obj.observation_nums_[label] = 1;

  // Add to global object registry
  objects_.push_back(obj);
  printFusionInfo(obj, label, "[New Object Cluster]");
}

void ObjectMap2D::mergeCellsIntoObjectCluster(const int& merged_object_id,
    const std::vector<Eigen::Vector2d>& new_cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;
  const auto last_objects = objects_;

  ObjectCluster& merged_object = objects_[merged_object_id];
  std::vector<Eigen::Vector2d> real_new_cells;

  // Process new spatial cells for integration
  for (auto new_cell : new_cells) {
    int adr = toAdr(new_cell);
    object_indexs_[adr] = merged_object_id;  // Associate with this object cluster

    // Add only genuinely new cells to avoid duplicates
    if (!merged_object.visited_.count(adr)) {
      real_new_cells.push_back(new_cell);
      merged_object.visited_[adr] = 1;
    }

    // Track observation frequency for high-confidence detections (label 0)
    if (label == 0) {
      if (merged_object.seen_counts_.count(adr))
        merged_object.seen_counts_[adr] += 1;
      else
        merged_object.seen_counts_[adr] = 1;

      // Update maximum observation count for this cluster
      if (merged_object.seen_counts_[adr] > merged_object.max_seen_count_)
        merged_object.max_seen_count_ = merged_object.seen_counts_[adr];
    }
  }

  // Extend cluster's spatial coverage
  merged_object.cells_.insert(
      merged_object.cells_.end(), real_new_cells.begin(), real_new_cells.end());

  // Update high-confidence cell tracking for label 0
  if (label == 0) {
    merged_object.good_cells_.clear();
    for (auto cell : merged_object.cells_) {
      int adr = toAdr(cell);
      if (merged_object.seen_counts_.count(adr)) {
        // Cells with sufficient observations are considered "good"
        if (merged_object.seen_counts_[adr] >= min(4, merged_object.max_seen_count_))
          merged_object.good_cells_.push_back(cell);
      }
    }
    ROS_ERROR("merged_object good cells size = %ld", merged_object.good_cells_.size());
  }

  // Recompute spatial properties
  merged_object.average_.setZero();
  merged_object.box_max2d_ = merged_object.cells_.front();
  merged_object.box_min2d_ = merged_object.cells_.front();
  for (auto cell : merged_object.cells_) {
    merged_object.average_ += cell;
    for (int i = 0; i < 2; ++i) {
      merged_object.box_min2d_[i] = min(merged_object.box_min2d_[i], cell[i]);
      merged_object.box_max2d_[i] = max(merged_object.box_max2d_[i], cell[i]);
    }
  }
  merged_object.average_ /= double(merged_object.cells_.size());

  // Handle point cloud fusion based on observation history
  if (!merged_object.observation_nums_[label]) {
    // First observation of this label - initialize directly
    merged_object.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_object.clouds_[label] = *(detected_object.cloud);
    merged_object.confidence_scores_[label] = detected_object.score;
    merged_object.observation_cloud_sums_[label] = detected_object.cloud->points.size();
    merged_object.observation_nums_[label] = 1;
    printFusionInfo(merged_object, label, "[New Label Merged]");
  }
  else {
    // Merge with existing observations using point cloud fusion
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_cloud = *(merged_object.clouds_[label]);  // Copy existing cloud
    *merged_cloud += *(detected_object.cloud);        // Add new observations

    // Apply voxel grid downsampling to manage point cloud size
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(merged_cloud);
    voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_filter.filter(*merged_cloud);
    merged_object.clouds_[label] = merged_cloud;

    // Update 3D bounding box from merged point cloud
    merged_object.box_max3d_ = Vector3d(merged_object.clouds_[label]->points[0].x,
        merged_object.clouds_[label]->points[0].y, merged_object.clouds_[label]->points[0].z);
    merged_object.box_min3d_ = Vector3d(merged_object.clouds_[label]->points[0].x,
        merged_object.clouds_[label]->points[0].y, merged_object.clouds_[label]->points[0].z);

    for (auto pt : merged_object.clouds_[label]->points) {
      Vector3d vec_pt = Vector3d(pt.x, pt.y, pt.z);
      for (int i = 0; i < 3; ++i) {
        merged_object.box_min3d_[i] = min(merged_object.box_min3d_[i], vec_pt[i]);
        merged_object.box_max3d_[i] = max(merged_object.box_max3d_[i], vec_pt[i]);
      }
    }

    // Update observation tracking
    merged_object.observation_nums_[label]++;
    merged_object.observation_cloud_sums_[label] += detected_object.cloud->points.size();

    // Prepare confidence fusion parameters
    int last_total = last_objects[merged_object_id].clouds_[label]->points.size();
    double last_total_confidence = last_objects[merged_object_id].confidence_scores_[label];
    int now_observation = detected_object.cloud->points.size();
    double now_confidence = detected_object.score;
    int now_total = merged_object.clouds_[label]->points.size();

    // Apply confidence fusion strategy based on fusion type
    if (fusion_type_ == 0)
      merged_object.confidence_scores_[label] = now_confidence;  // Replace with current
    else if (fusion_type_ == 1)
      merged_object.confidence_scores_[label] =
          fusionConfidenceScore(last_total, last_total_confidence, now_observation, now_confidence,
              now_total, merged_object.observation_cloud_sums_[label]);  // Weighted fusion
    else if (fusion_type_ == 2)
      merged_object.confidence_scores_[label] =
          max(merged_object.confidence_scores_[label], now_confidence);  // Maximum confidence
    printFusionInfo(merged_object, label, "[Fusion]");
  }
}

/**
 * @brief Fusion algorithm for combining confidence scores from multiple observations
 *
 * This function implements a weighted confidence fusion strategy that combines
 * historical confidence with new observations, considering both the quantity
 * of evidence and the quality of individual detections.
 *
 * @param total_num_last Number of points in previous observation
 * @param c_last Previous confidence score
 * @param n_num_now Number of points in current observation
 * @param c_now Current confidence score
 * @param total_now Total points after fusion
 * @param sum Cumulative observation count
 * @return Fused confidence score combining all evidence
 */
double ObjectMap2D::fusionConfidenceScore(
    int total_num_last, double c_last, int n_num_now, double c_now, int total_now, int sum)
{
  double n_now = (double)n_num_now;
  double w_last, w_now, final_score;
  // Calculate weighted fusion based on observation counts
  w_last = (sum - n_now) / sum;                   // Weight for historical evidence
  w_now = n_now / sum;                            // Weight for current observation
  final_score = w_last * c_last + w_now * c_now;  // Weighted combination
  return final_score;
}

bool ObjectMap2D::checkSafety(const Eigen::Vector2i& idx)
{
  if (sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN ||
      sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED || sdf_map_->getInflateOccupancy(idx) == 1)
    return false;
  return true;
}

bool ObjectMap2D::checkSafety(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return checkSafety(idx);
}

void ObjectMap2D::updateExposureHeatmap(
    int object_id, int observed_label, const vector<Eigen::Vector2d>& observed_cells)
{
  if (!exposure_heatmap_enabled_ || object_id < 0 || object_id >= (int)objects_.size())
    return;

  ObjectCluster& object = objects_[object_id];
  for (const auto& cell : object.cells_)
    object.exposure_by_grid_.emplace(toAdr(cell), 0.0);
  std::unordered_map<int, Vector2d> unique_cells;
  for (const auto& cell : observed_cells)
    unique_cells.emplace(toAdr(cell), cell);

  ExposureHeatmapUpdate update;
  update.scene_id = exposure_scene_id_;
  update.episode_id = exposure_episode_id_;
  update.step_index = exposure_step_index_;
  update.cluster_id = object.id_;
  update.observed_label = observed_label;
  update.best_label = object.best_label_;
  update.camera_position = exposure_camera_position_;
  update.camera_yaw = exposure_camera_yaw_;

  const double half_fov = exposure_hfov_rad_ * 0.5;
  for (const auto& entry : unique_cells) {
    const Vector2d& cell = entry.second;
    const Vector2d delta = cell - exposure_camera_position_.head<2>();
    const double angle = atan2(delta.y(), delta.x()) - exposure_camera_yaw_;
    const double wrapped = atan2(sin(angle), cos(angle));
    if (fabs(wrapped) > half_fov)
      continue;

    const double raw_weight = cos(M_PI * fabs(wrapped) / (2.0 * half_fov));
    const double before = object.exposure_by_grid_[entry.first];
    const double after = min(exposure_capacity_, before + max(0.0, raw_weight));
    object.exposure_by_grid_[entry.first] = after;
    update.grid_addresses.push_back(entry.first);
    update.exposure_before.push_back(before);
    update.exposure_after.push_back(after);
  }

  if (update.grid_addresses.empty())
    return;

  double total = 0.0;
  int saturated = 0;
  for (const auto& entry : object.exposure_by_grid_) {
    total += entry.second;
    if (entry.second >= exposure_capacity_ - 1e-6)
      ++saturated;
  }
  update.total_exposure = total;
  update.mean_exposure = total / object.exposure_by_grid_.size();
  update.saturation_ratio =
      static_cast<float>(saturated) / static_cast<float>(object.exposure_by_grid_.size());
  pending_exposure_updates_.push_back(update);
  exposure_rank_pending_ = true;
  ROS_INFO("[ExposureHeatmap][UPDATE] scene=%s episode=%s step=%u cluster=%d observed_label=%d "
           "best_label=%d cells=%zu total=%.3f mean=%.3f saturated=%.3f pose=(%.2f,%.2f) yaw=%.2f",
      update.scene_id.c_str(), update.episode_id.c_str(), update.step_index, update.cluster_id,
      update.observed_label, update.best_label, update.grid_addresses.size(), update.total_exposure,
      update.mean_exposure, update.saturation_ratio, update.camera_position.x(),
      update.camera_position.y(), update.camera_yaw);
}

vector<ExposureViewCandidate> ObjectMap2D::consumeExposureViewCandidates(int max_candidates)
{
  vector<ExposureViewCandidate> candidates;
  if (!exposure_rank_pending_ || max_candidates <= 0)
    return candidates;
  exposure_rank_pending_ = false;

  for (const auto& object : objects_) {
    vector<pair<double, Vector2d>> boundary;
    for (const auto& cell : object.cells_) {
      Eigen::Vector2i idx;
      sdf_map_->posToIndex(cell, idx);
      bool external = false;
      for (const auto& neighbor : fourNeighbors(idx)) {
        if (!inMap(neighbor) || object_indexs_[toAdr(neighbor)] != object.id_) {
          external = true;
          break;
        }
      }
      if (!external)
        continue;
      const int address = toAdr(cell);
      const auto exposure = object.exposure_by_grid_.find(address);
      const double value = exposure == object.exposure_by_grid_.end() ? 0.0 : exposure->second;
      boundary.push_back(make_pair(exposure_capacity_ - value, cell));
    }
    sort(boundary.begin(), boundary.end(),
        [](const pair<double, Vector2d>& a, const pair<double, Vector2d>& b) { return a.first > b.first; });

    for (const auto& entry : boundary) {
      if ((int)candidates.size() >= max_candidates)
        break;
      Vector2d outward = entry.second - object.average_;
      if (outward.norm() < 1e-4)
        outward = Vector2d(1.0, 0.0);
      outward.normalize();
      for (const double distance : {0.50, 0.70, 0.85}) {
        if ((int)candidates.size() >= max_candidates)
          break;
        const Vector2d parking = entry.second + distance * outward;
        if (!sdf_map_->isInMap(parking) || !checkSafety(parking))
          continue;
        ExposureViewCandidate candidate;
        candidate.context.scene_id = exposure_scene_id_;
        candidate.context.episode_id = exposure_episode_id_;
        candidate.context.step_index = exposure_step_index_;
        candidate.context.cluster_id = object.id_;
        candidate.context.observed_label = object.best_label_;
        candidate.context.best_label = object.best_label_;
        candidate.context.camera_position = exposure_camera_position_;
        candidate.context.camera_yaw = exposure_camera_yaw_;
        candidate.context.total_exposure = 0.0;
        candidate.context.mean_exposure = 0.0;
        candidate.context.saturation_ratio = 0.0;
        candidate.position = parking;
        candidate.yaw = atan2(entry.second.y() - parking.y(), entry.second.x() - parking.x());
        candidate.gain = max(0.0, entry.first);
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

Eigen::Vector3d ObjectMap2D::exposureColor(int label, double normalized_exposure) const
{
  static const Eigen::Vector3d colors[] = {
      Eigen::Vector3d(0.85, 0.10, 0.10), Eigen::Vector3d(0.10, 0.62, 0.20),
      Eigen::Vector3d(0.10, 0.32, 0.85), Eigen::Vector3d(0.92, 0.45, 0.08),
      Eigen::Vector3d(0.55, 0.18, 0.72)};
  const Eigen::Vector3d base = colors[(label < 0 ? 0 : label) % 5];
  const double level = max(0.0, min(1.0, normalized_exposure));
  return (1.0 - level) * Eigen::Vector3d::Ones() + level * base;
}

void ObjectMap2D::publishExposureHeatmap()
{
  if (!exposure_heatmap_enabled_)
    return;
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  for (const auto& object : objects_) {
    for (const auto& entry : object.exposure_by_grid_) {
      Eigen::Vector2i idx;
      Vector2d pos;
      idx = sdf_map_->addressToIdx(entry.first);
      sdf_map_->indexToPos(idx, pos);
      const Eigen::Vector3d color = exposureColor(object.best_label_, entry.second / exposure_capacity_);
      pcl::PointXYZRGB point;
      point.x = pos.x();
      point.y = pos.y();
      point.z = 0.12;
      point.r = static_cast<uint8_t>(255 * color.x());
      point.g = static_cast<uint8_t>(255 * color.y());
      point.b = static_cast<uint8_t>(255 * color.z());
      cloud.points.push_back(point);
    }
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(cloud, output);
  output.header.frame_id = "world";
  output.header.stamp = ros::Time::now();
  exposure_heatmap_pub_.publish(output);
  ROS_INFO_THROTTLE(1.0, "[ExposureHeatmap][PUBLISH] topic=/object/exposure_heatmap clusters=%zu points=%zu",
      objects_.size(), cloud.points.size());
}

void ObjectMap2D::getObjects(
    vector<vector<Eigen::Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels)
{
  clusters.clear();
  averages.clear();
  labels.clear();
  for (auto object : objects_) {
    clusters.push_back(object.cells_);
    averages.push_back(object.average_);
    labels.push_back(object.best_label_);
  }
}

void ObjectMap2D::getObjectBoxes(vector<pair<Eigen::Vector2d, Eigen::Vector2d>>& boxes)
{
  boxes.clear();
  for (auto object : objects_) {
    Vector2d center = (object.box_max2d_ + object.box_min2d_) * 0.5;
    Vector2d scale = object.box_max2d_ - object.box_min2d_;
    boxes.push_back(make_pair(center, scale));
  }
}

void ObjectMap2D::getObjectBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes)
{
  boxes.clear();
  for (auto object : objects_) {
    Vector3d center = (object.box_max3d_ + object.box_min3d_) * 0.5;
    Vector3d scale = object.box_max3d_ - object.box_min3d_;
    boxes.push_back(make_pair(center, scale));
  }
}

void ObjectMap2D::getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax)
{
  bmin.clear();
  bmax.clear();
  for (auto object : objects_) {
    bmin.push_back(object.box_min3d_);
    bmax.push_back(object.box_max3d_);
  }
}

void ObjectMap2D::getAllConfidenceObjectClouds(
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_clouds)
{
  object_clouds.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // Extract high-confidence object cells
  for (auto object : objects_) {
    if (object.confidence_scores_[0] >= min_confidence_) {
      for (auto cell : object.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell[0];
        point.y = cell[1];
        point.z = 0;
        object_clouds->push_back(point);
      }
    }
  }
}

void ObjectMap2D::getTopConfidenceObjectCloud(
    vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& top_object_clouds,
    bool limited_confidence, bool extreme)
{
  top_object_clouds.clear();
  vector<ObjectCluster> top_objects;

  // Confidence filtering strategy
  // TODO: May need logic adjustment for relaxed no limited_confidence conditions
  if (!limited_confidence) {
    // Include all objects without confidence filtering
    for (auto object : objects_) top_objects.push_back(object);

    // Sort by confidence score in descending order
    std::sort(
        top_objects.begin(), top_objects.end(), [](const ObjectCluster& a, const ObjectCluster& b) {
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds for top-ranked objects
    for (auto top_obj : top_objects) {
      if (top_obj.confidence_scores_[0] <= 0.01)
        break;  // Skip extremely low confidence objects

      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud;
      top_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      for (auto cell : top_obj.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell(0);
        point.y = cell(1);
        point.z = 0;
        top_object_cloud->push_back(point);
      }
      top_object_clouds.push_back(top_object_cloud);
    }

    // Fallback for extreme mode when no high-confidence objects exist
    if (extreme && top_object_clouds.empty()) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> others_object_cloud;
      others_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      // Include all object cells regardless of confidence
      for (auto object : objects_) {
        for (auto cell : object.cells_) {
          pcl::PointXYZ point;
          point.x = cell(0);
          point.y = cell(1);
          point.z = 0;
          others_object_cloud->push_back(point);
        }
      }
      top_object_clouds.push_back(others_object_cloud);
    }
  }
  else {
    // Apply confidence filtering with functional scoring
    for (auto object : objects_) {
      int max_func_score = 0, best_label = -1;

      // Find best label using functional score (observation count * confidence)
      for (int label = 0; label < (int)object.clouds_.size(); label++) {
        auto obs_sum = object.observation_cloud_sums_[label];
        auto score = object.confidence_scores_[label];
        int func_score = obs_sum * score;
        if (func_score > max_func_score) {
          max_func_score = func_score;
          best_label = label;
        }
      }

      // Include only high-confidence objects with primary label (0)
      if (best_label == 0 && isConfidenceObject(object))
        top_objects.push_back(object);
    }

    // Sort filtered objects by confidence
    std::sort(
        top_objects.begin(), top_objects.end(), [](const ObjectCluster& a, const ObjectCluster& b) {
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds from filtered objects
    for (auto top_obj : top_objects) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud;
      top_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
      for (auto cell : top_obj.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell(0);
        point.y = cell(1);
        point.z = 0;
        top_object_cloud->push_back(point);
      }
      top_object_clouds.push_back(top_object_cloud);
    }
  }
}

bool ObjectMap2D::isConfidenceObject(const ObjectCluster& obj)
{
  if (obj.confidence_scores_[0] >= min_confidence_ &&
      obj.observation_nums_[0] >= min_observation_num_)
    return true;
  return false;
}

}  // namespace apexnav_planner
