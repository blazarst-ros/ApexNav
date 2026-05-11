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
#include <plan_env/semantic_observability.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace apexnav_planner {
ObjectMap2D::ObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
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
  nh.param("object/use_semantic_observability", use_semantic_observability_, true);
  nh.param("object/vis_cloud", is_vis_cloud_, false);
  nh.param("object/lambda_d", lambda_d_, 0.25);
  nh.param("object/r0", r0_, 0.013);
  nh.param("object/mask_sigmoid_k", mask_sigmoid_k_, 300.0);
  nh.param("object/beta", beta_, 0.5);
  nh.param("object/min_semantic_evidence", min_semantic_evidence_, 0.05);
  nh.param("object/verification_margin_threshold", verification_margin_threshold_, 0.12);
  nh.param("object/verification_accept_margin", verification_accept_margin_, 0.18);
  nh.param("object/verification_max_count", verification_max_count_, 1);

  // Setup ROS communication
  object_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object/clouds", 10);

  // Configure raycasting for spatial queries
  raycaster_.reset(new RayCaster2D);
  resolution_ = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  // Set point cloud processing parameters
  leaf_size_ = 0.1f;  // Voxel grid leaf size for downsampling
}

void ObjectMap2D::reset()
{
  objects_.clear();
  fill(object_buffer_.begin(), object_buffer_.end(), 0);
  fill(object_indexs_.begin(), object_indexs_.end(), -1);
  all_object_clouds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  over_depth_object_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
}

void ObjectMap2D::setConfidenceThreshold(double val)
{
  min_confidence_ = val;
  ROS_INFO("Set Confidence Threshold = %f", val);
}

bool ObjectMap2D::getSemanticEvidenceSnapshot(
    int object_id, int label, SemanticEvidenceSnapshot& snapshot)
{
  if (object_id < 0 || object_id >= (int)objects_.size())
    return false;
  const ObjectCluster& object = objects_[object_id];
  const int evidence_label_count = std::min({ (int)object.quality_evidence_scores_.size(),
      (int)object.confidence_scores_.size(), (int)object.observation_nums_.size(),
      (int)object.observation_cloud_sums_.size(), (int)object.observability_scores_.size(),
      (int)object.last_distances_.size(), (int)object.last_view_angles_.size(),
      (int)object.last_mask_scales_.size() });
  if (label < 0 || label >= evidence_label_count)
    return false;

  snapshot.cluster_id = object_id;
  snapshot.label = label;
  snapshot.best_label = object.best_label_;
  snapshot.use_semantic_observability = use_semantic_observability_;
  snapshot.lambda_d = lambda_d_;
  snapshot.r0 = r0_;
  snapshot.mask_sigmoid_k = mask_sigmoid_k_;
  snapshot.beta = beta_;
  snapshot.min_semantic_evidence = min_semantic_evidence_;
  snapshot.min_observation_num = min_observation_num_;
  snapshot.fused_confidence = object.confidence_scores_[label];
  snapshot.observation_num = object.observation_nums_[label];
  snapshot.observation_cloud_sum = object.observation_cloud_sums_[label];
  snapshot.observability = object.observability_scores_[label];
  snapshot.quality_evidence = object.quality_evidence_scores_[label];

  if (evidence_label_count > 0) {
    snapshot.target_quality_evidence = object.quality_evidence_scores_[0];
    snapshot.target_fused_confidence = object.confidence_scores_[0];
    snapshot.target_observation_num = object.observation_nums_[0];
    snapshot.target_passes_threshold =
        object.quality_evidence_scores_[0] >= min_semantic_evidence_ &&
        object.observation_nums_[0] >= min_observation_num_;
    snapshot.target_is_best_label = object.best_label_ == 0;
  }

  int top_label = -1;
  int second_label = -1;
  double top_score = -std::numeric_limits<double>::infinity();
  double second_score = -std::numeric_limits<double>::infinity();
  for (int semantic_label = 0; semantic_label < evidence_label_count; ++semantic_label) {
    const double score = object.quality_evidence_scores_[semantic_label];
    if (score > top_score) {
      second_score = top_score;
      second_label = top_label;
      top_score = score;
      top_label = semantic_label;
    }
    else if (score > second_score) {
      second_score = score;
      second_label = semantic_label;
    }
  }

  auto fillEvidenceRank = [&](int semantic_label, bool is_top) {
    if (semantic_label < 0)
      return;
    const bool passes_threshold =
        object.quality_evidence_scores_[semantic_label] >= min_semantic_evidence_ &&
        object.observation_nums_[semantic_label] >= min_observation_num_;
    if (is_top) {
      snapshot.top_label = semantic_label;
      snapshot.top_quality_evidence = object.quality_evidence_scores_[semantic_label];
      snapshot.top_fused_confidence = object.confidence_scores_[semantic_label];
      snapshot.top_observation_num = object.observation_nums_[semantic_label];
      snapshot.top_observation_cloud_sum = object.observation_cloud_sums_[semantic_label];
      snapshot.top_observability = object.observability_scores_[semantic_label];
      snapshot.top_distance = object.last_distances_[semantic_label];
      snapshot.top_view_angle = object.last_view_angles_[semantic_label];
      snapshot.top_mask_scale = object.last_mask_scales_[semantic_label];
      snapshot.top_passes_threshold = passes_threshold;
    }
    else {
      snapshot.second_label = semantic_label;
      snapshot.second_quality_evidence = object.quality_evidence_scores_[semantic_label];
      snapshot.second_fused_confidence = object.confidence_scores_[semantic_label];
      snapshot.second_observation_num = object.observation_nums_[semantic_label];
      snapshot.second_observation_cloud_sum = object.observation_cloud_sums_[semantic_label];
      snapshot.second_observability = object.observability_scores_[semantic_label];
      snapshot.second_distance = object.last_distances_[semantic_label];
      snapshot.second_view_angle = object.last_view_angles_[semantic_label];
      snapshot.second_mask_scale = object.last_mask_scales_[semantic_label];
      snapshot.second_passes_threshold = passes_threshold;
    }
  };

  fillEvidenceRank(top_label, true);
  fillEvidenceRank(second_label, false);
  snapshot.top_second_abs_diff =
      second_label >= 0 ? std::abs(snapshot.top_quality_evidence -
                              snapshot.second_quality_evidence)
                        : snapshot.top_quality_evidence;
  snapshot.target_is_top_label = snapshot.top_label == 0;

  return true;
}

void ObjectMap2D::getSemanticEvidenceConfig(SemanticEvidenceSnapshot& snapshot) const
{
  snapshot.use_semantic_observability = use_semantic_observability_;
  snapshot.lambda_d = lambda_d_;
  snapshot.r0 = r0_;
  snapshot.mask_sigmoid_k = mask_sigmoid_k_;
  snapshot.beta = beta_;
  snapshot.min_semantic_evidence = min_semantic_evidence_;
  snapshot.min_observation_num = min_observation_num_;
}

void ObjectMap2D::getVerificationCandidates(std::vector<VerificationCandidate>& candidates)
{
  candidates.clear();

  double target_mu_v = 1.0;
  double target_sigma_v = 0.35;
  ros::param::param("/semantic_prior/label_0/mu_v", target_mu_v, target_mu_v);
  ros::param::param("/semantic_prior/label_0/sigma_v", target_sigma_v, target_sigma_v);

  for (auto& object : objects_) {
    updateVerificationState(object);
    if (object.verification_verified_ ||
        (!object.verification_pending_ &&
            object.verification_count_ >= verification_max_count_) ||
        object.quality_evidence_scores_.size() < 2 ||
        object.observation_nums_.empty()) {
      continue;
    }

    int top_label = -1;
    int second_label = -1;
    double top_score = -std::numeric_limits<double>::infinity();
    double second_score = -std::numeric_limits<double>::infinity();
    const int label_count = (int)object.quality_evidence_scores_.size();
    for (int label = 0; label < label_count; ++label) {
      const double score = object.quality_evidence_scores_[label];
      if (score > top_score) {
        second_score = top_score;
        second_label = top_label;
        top_score = score;
        top_label = label;
      }
      else if (score > second_score) {
        second_score = score;
        second_label = label;
      }
    }

    if (top_label != 0 || second_label < 0)
      continue;
    if (top_score < min_semantic_evidence_ || object.observation_nums_[0] < min_observation_num_)
      continue;

    const double margin = top_score - second_score;
    if (margin >= verification_margin_threshold_)
      continue;

    VerificationCandidate candidate;
    candidate.object_id = object.id_;
    candidate.position = object.average_;
    candidate.top_score = top_score;
    candidate.second_score = second_score;
    candidate.margin = margin;
    candidate.target_mu_v = target_mu_v;
    candidate.target_sigma_v = target_sigma_v;
    candidate.source_agent_id = object.last_observer_agent_;
    candidate.verify_count = object.verification_count_;
    candidate.verified = object.verification_verified_;
    candidate.object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    const auto& source_cells = object.good_cells_.empty() ? object.cells_ : object.good_cells_;
    for (const auto& cell : source_cells) {
      pcl::PointXYZ point;
      point.x = cell(0);
      point.y = cell(1);
      point.z = 0.0;
      candidate.object_cloud->push_back(point);
    }
    if (candidate.object_cloud->points.empty())
      continue;

    if (!object.verification_pending_) {
      object.verification_pending_ = true;
      object.verification_count_++;
    }
    candidate.verify_count = object.verification_count_;
    candidates.push_back(candidate);
  }
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

  const int object_count = std::min((int)observation_clouds.size(), (int)objects_.size());

  // Process each observation cloud against corresponding objects
  for (int i = 0; i < object_count; i++) {
    auto observation_cloud = observation_clouds[i];
    auto object = objects_[i];

    if (!observation_cloud || observation_cloud->points.empty())
      continue;

    // Check overlap with each possible object classification
    for (int label = 0; label < (int)object.confidence_scores_.size(); ++label) {
      if (object.confidence_scores_[label] < 1e-3 || !object.clouds_[label])
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
      merged_object.quality_evidence_scores_[label] =
          semantic_observability::saturatedEvidence(merged_object.observability_scores_[label],
              merged_object.confidence_scores_[label], merged_object.observation_nums_[label], beta_);
      printFusionInfo(merged_object, label, "[Observation]");
      // ROS_WARN("[Observation] id = %d label = %d overlap_count = %d object_cloud = %ld",
      //     merged_object.id_, label, overlap_count, object.clouds_[label]->points.size());
    }
  }
}

int ObjectMap2D::searchSingleObjectCluster(const DetectedObject& detected_object)
{
  auto object_cloud = detected_object.cloud;
  if (!object_cloud || object_cloud->points.empty()) {
    return -1;
  }

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
    if (!sdf_map_->isInMap(pt_w))
      continue;

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
      if (!sdf_map_->isInMap(nbr))
        continue;
      int nbr_adr = sdf_map_->toAddress(nbr);
      if (object_indexs_[nbr_adr] != -1) {
        const int candidate_idx = object_indexs_[nbr_adr];
        if (candidate_idx < 0 || candidate_idx >= (int)objects_.size()) {
          ROS_WARN_THROTTLE(1.0,
              "[ObjectMap2D] Drop stale object index %d at grid address %d", candidate_idx,
              nbr_adr);
          object_indexs_[nbr_adr] = -1;
          continue;
        }
        // Found existing object cluster - use first match
        // TODO: Implement multi-object merging for complex scenarios
        obj_idx = candidate_idx;
        break;
      }
    }
    if (obj_idx != -1)
      break;
  }

  // Apply voxel grid filtering to reduce point cloud density
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(object_cloud);
  voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  voxel_filter.filter(*detected_object.cloud);
  if (!detected_object.cloud || detected_object.cloud->points.empty()) {
    ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore object whose cloud became empty after filtering");
    return -1;
  }

  // Either merge with existing cluster or create new one
  if (obj_idx != -1) {
    mergeCellsIntoObjectCluster(obj_idx, object_point2Ds, detected_object);
  }
  else {
    createNewObjectCluster(object_point2Ds, detected_object);
    obj_idx = object_indexs_[toAdr(object_point2Ds[0])];
  }

  // Update classification and visualization
  if (obj_idx < 0 || obj_idx >= (int)objects_.size()) {
    ROS_ERROR("[ObjectMap2D] Failed to create or find a valid object cluster");
    return -1;
  }

  updateObjectBestLabel(obj_idx);
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
  double max_func_score = use_semantic_observability_ ? min_semantic_evidence_ : 0.1;
  int best_label = -1;

  // Evaluate each possible classification label
  for (int label = 0; label < (int)objects_[obj_idx].clouds_.size(); label++) {
    auto obs_sum = objects_[obj_idx].observation_cloud_sums_[label];
    auto score = objects_[obj_idx].confidence_scores_[label];
    double func_score = use_semantic_observability_
                            ? objects_[obj_idx].quality_evidence_scores_[label]
                            : obs_sum * score;

    if (func_score > max_func_score) {
      max_func_score = func_score;
      best_label = label;
    }
  }
  objects_[obj_idx].best_label_ = best_label;
  updateVerificationState(objects_[obj_idx]);
}

void ObjectMap2D::updateVerificationState(ObjectCluster& object)
{
  if (!object.verification_pending_ || object.verification_verified_ ||
      object.quality_evidence_scores_.size() < 2) {
    return;
  }

  int top_label = -1;
  int second_label = -1;
  double top_score = -std::numeric_limits<double>::infinity();
  double second_score = -std::numeric_limits<double>::infinity();
  for (int label = 0; label < (int)object.quality_evidence_scores_.size(); ++label) {
    const double score = object.quality_evidence_scores_[label];
    if (score > top_score) {
      second_score = top_score;
      second_label = top_label;
      top_score = score;
      top_label = label;
    }
    else if (score > second_score) {
      second_score = score;
      second_label = label;
    }
  }

  if (top_label == 0 && second_label >= 0 &&
      top_score - second_score >= verification_accept_margin_) {
    object.verification_verified_ = true;
    object.verification_pending_ = false;
  }
}

void ObjectMap2D::updateQualityAwareEvidence(
    ObjectCluster& object, int label, const DetectedObject& detected_object)
{
  if (label < 0 || label >= (int)object.quality_evidence_scores_.size())
    return;

  double rho = semantic_observability::observability(detected_object.camera_height,
      detected_object.mu_v, detected_object.sigma_v, detected_object.distance,
      detected_object.view_angle, detected_object.mask_scale, lambda_d_, r0_, mask_sigmoid_k_);
  double evidence = semantic_observability::saturatedEvidence(
      rho, object.confidence_scores_[label], object.observation_nums_[label], beta_);

  object.observability_scores_[label] = rho;
  object.quality_evidence_scores_[label] = evidence;
  object.last_distances_[label] = detected_object.distance;
  object.last_view_angles_[label] = detected_object.view_angle;
  object.last_mask_scales_[label] = detected_object.mask_scale;
}

void ObjectMap2D::ensureObjectLabelCapacity(ObjectCluster& object, int label)
{
  if (label < 0 || label < (int)object.confidence_scores_.size())
    return;

  const size_t new_size = (size_t)label + 1;
  object.clouds_.resize(new_size);
  object.confidence_scores_.resize(new_size, 0.0);
  object.observability_scores_.resize(new_size, 0.0);
  object.quality_evidence_scores_.resize(new_size, 0.0);
  object.last_distances_.resize(new_size, 0.0);
  object.last_view_angles_.resize(new_size, 0.0);
  object.last_mask_scales_.resize(new_size, 0.0);
  object.observation_nums_.resize(new_size, 0);
  object.observation_cloud_sums_.resize(new_size, 0);
}

bool ObjectMap2D::updateObject3DBounds(ObjectCluster& object, int label)
{
  if (label < 0 || label >= (int)object.clouds_.size() || !object.clouds_[label] ||
      object.clouds_[label]->points.empty()) {
    return false;
  }

  const auto& first_point = object.clouds_[label]->points.front();
  object.box_max3d_ = Vector3d(first_point.x, first_point.y, first_point.z);
  object.box_min3d_ = object.box_max3d_;

  for (const auto& pt : object.clouds_[label]->points) {
    Vector3d vec_pt(pt.x, pt.y, pt.z);
    for (int i = 0; i < 3; ++i) {
      object.box_min3d_[i] = min(object.box_min3d_[i], vec_pt[i]);
      object.box_max3d_[i] = max(object.box_max3d_[i], vec_pt[i]);
    }
  }

  return true;
}

void ObjectMap2D::createNewObjectCluster(
    const std::vector<Eigen::Vector2d>& cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;
  if (label < 0) {
    ROS_WARN("[ObjectMap2D] Ignore detected object with invalid label %d", label);
    return;
  }
  if (cells.empty() || !detected_object.cloud || detected_object.cloud->points.empty()) {
    ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore empty object cluster candidate");
    return;
  }

  // Initialize new object cluster with unique ID
  ObjectCluster obj(std::max(6, label + 1));
  obj.id_ = (int)objects_.size();
  obj.max_seen_count_ = 0;
  obj.good_cells_.clear();
  obj.seen_counts_.clear();
  obj.best_label_ = -1;
  obj.last_observer_agent_ = detected_object.agent_id;

  // Process spatial cells and establish grid associations
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
  if (!updateObject3DBounds(obj, label)) {
    ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore object with empty cloud bounds");
    return;
  }

  // Initialize confidence tracking for this object
  obj.confidence_scores_[label] = detected_object.score;
  obj.observation_cloud_sums_[label] = detected_object.cloud->points.size();
  obj.observation_nums_[label] = 1;
  updateQualityAwareEvidence(obj, label, detected_object);

  // Add to global object registry
  objects_.push_back(obj);
  printFusionInfo(obj, label, "[New Object Cluster]");
}

void ObjectMap2D::mergeCellsIntoObjectCluster(const int& merged_object_id,
    const std::vector<Eigen::Vector2d>& new_cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;
  if (label < 0) {
    ROS_WARN("[ObjectMap2D] Ignore detected object with invalid label %d", label);
    return;
  }
  if (merged_object_id < 0 || merged_object_id >= (int)objects_.size()) {
    ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore stale merge target id %d", merged_object_id);
    return;
  }
  if (!detected_object.cloud || detected_object.cloud->points.empty()) {
    ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore empty object cloud for merge");
    return;
  }

  ensureObjectLabelCapacity(objects_[merged_object_id], label);
  const auto last_objects = objects_;

  ObjectCluster& merged_object = objects_[merged_object_id];
  merged_object.last_observer_agent_ = detected_object.agent_id;
  std::vector<Eigen::Vector2d> real_new_cells;

  // Process new spatial cells for integration
  for (auto new_cell : new_cells) {
    if (!sdf_map_->isInMap(new_cell))
      continue;
    int adr = toAdr(new_cell);
    if (adr < 0 || adr >= (int)object_indexs_.size())
      continue;
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
    ROS_DEBUG("merged_object good cells size = %ld", merged_object.good_cells_.size());
  }

  // Recompute spatial properties
  if (merged_object.cells_.empty()) {
    ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Merge target %d has no valid cells", merged_object_id);
    return;
  }

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
    if (!updateObject3DBounds(merged_object, label)) {
      ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore first label merge with empty cloud bounds");
      return;
    }
    merged_object.confidence_scores_[label] = detected_object.score;
    merged_object.observation_cloud_sums_[label] = detected_object.cloud->points.size();
    merged_object.observation_nums_[label] = 1;
    updateQualityAwareEvidence(merged_object, label, detected_object);
    printFusionInfo(merged_object, label, "[New Label Merged]");
  }
  else {
    if (!merged_object.clouds_[label] || merged_object.clouds_[label]->points.empty()) {
      merged_object.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
      *merged_object.clouds_[label] = *(detected_object.cloud);
      if (!updateObject3DBounds(merged_object, label)) {
        ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Ignore merge with empty replacement cloud");
        return;
      }
      merged_object.confidence_scores_[label] = detected_object.score;
      merged_object.observation_cloud_sums_[label] = detected_object.cloud->points.size();
      merged_object.observation_nums_[label] = 1;
      updateQualityAwareEvidence(merged_object, label, detected_object);
      printFusionInfo(merged_object, label, "[Recovered Label Merged]");
      return;
    }

    // Merge with existing observations using point cloud fusion
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_cloud = *(merged_object.clouds_[label]);  // Copy existing cloud
    *merged_cloud += *(detected_object.cloud);        // Add new observations

    // Apply voxel grid downsampling to manage point cloud size
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(merged_cloud);
    voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_filter.filter(*merged_cloud);
    if (merged_cloud->points.empty()) {
      ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Skip merge that produced an empty cloud");
      return;
    }
    merged_object.clouds_[label] = merged_cloud;

    // Update 3D bounding box from merged point cloud
    if (!updateObject3DBounds(merged_object, label)) {
      ROS_WARN_THROTTLE(1.0, "[ObjectMap2D] Skip merge with invalid cloud bounds");
      return;
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
    updateQualityAwareEvidence(merged_object, label, detected_object);
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
    if (object.verification_pending_ && !object.verification_verified_)
      continue;
    if (object.quality_evidence_scores_.empty() || object.confidence_scores_.empty() ||
        object.observation_nums_.empty())
      continue;
    bool score_ok = use_semantic_observability_
                        ? object.quality_evidence_scores_[0] >= min_semantic_evidence_
                        : object.confidence_scores_[0] >= min_confidence_;
    if (score_ok) {
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
    for (auto object : objects_) {
      if (object.verification_pending_ && !object.verification_verified_)
        continue;
      if (object.quality_evidence_scores_.empty() || object.confidence_scores_.empty() ||
          object.observation_nums_.empty())
        continue;
      top_objects.push_back(object);
    }

    // Sort by confidence score in descending order
    std::sort(top_objects.begin(), top_objects.end(),
        [this](const ObjectCluster& a, const ObjectCluster& b) {
          if (use_semantic_observability_)
            return a.quality_evidence_scores_[0] > b.quality_evidence_scores_[0];
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds for top-ranked objects
    for (auto top_obj : top_objects) {
      double object_score = use_semantic_observability_ ? top_obj.quality_evidence_scores_[0]
                                                        : top_obj.confidence_scores_[0];
      double min_score = use_semantic_observability_ ? min_semantic_evidence_ : 0.01;
      if (object_score <= min_score)
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
      if (!top_object_cloud->points.empty())
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
      if (!others_object_cloud->points.empty())
        top_object_clouds.push_back(others_object_cloud);
    }
  }
  else {
    // Apply confidence filtering with functional scoring
    for (auto object : objects_) {
      if (object.verification_pending_ && !object.verification_verified_)
        continue;
      if (object.quality_evidence_scores_.empty() || object.confidence_scores_.empty() ||
          object.observation_nums_.empty())
        continue;
      double max_func_score = use_semantic_observability_ ? min_semantic_evidence_ : 0.0;
      int best_label = -1;

      // Find best label using functional score (observation count * confidence)
      const int label_count = std::min({ (int)object.clouds_.size(),
          (int)object.observation_cloud_sums_.size(), (int)object.confidence_scores_.size(),
          (int)object.quality_evidence_scores_.size() });
      for (int label = 0; label < label_count; label++) {
        auto obs_sum = object.observation_cloud_sums_[label];
        auto score = object.confidence_scores_[label];
        double func_score = use_semantic_observability_
                                ? object.quality_evidence_scores_[label]
                                : obs_sum * score;
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
    std::sort(top_objects.begin(), top_objects.end(),
        [this](const ObjectCluster& a, const ObjectCluster& b) {
          if (use_semantic_observability_)
            return a.quality_evidence_scores_[0] > b.quality_evidence_scores_[0];
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
      if (!top_object_cloud->points.empty())
        top_object_clouds.push_back(top_object_cloud);
    }
  }
}

bool ObjectMap2D::isConfidenceObject(const ObjectCluster& obj)
{
  if (obj.verification_pending_ && !obj.verification_verified_)
    return false;
  if (obj.quality_evidence_scores_.empty() || obj.confidence_scores_.empty() ||
      obj.observation_nums_.empty())
    return false;

  bool score_ok = use_semantic_observability_
                      ? obj.quality_evidence_scores_[0] >= min_semantic_evidence_
                      : obj.confidence_scores_[0] >= min_confidence_;
  if (score_ok && obj.observation_nums_[0] >= min_observation_num_)
    return true;
  return false;
}

}  // namespace apexnav_planner
