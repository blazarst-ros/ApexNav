/**
 * @file map_ros.cpp
 * @brief Implementation of ROS interface for 2D SDF mapping system
 *
 * This file implements the MapROS class which provides the ROS interface for
 * the 2D signed distance field mapping system. It handles sensor data processing,
 * object detection integration, and real-time map visualization.
 * 
 * @author Zager-Zhang
 */

#include <plan_env/map_ros.h>
#include <plan_env/semantic_cloud_contract.h>

#include <stdexcept>

namespace apexnav_planner {

namespace {

bool isFiniteCameraCalibration(const sensor_msgs::CameraInfo& info)
{
  for (const double value : info.K)
    if (!std::isfinite(value))
      return false;
  for (const double value : info.D)
    if (!std::isfinite(value))
      return false;
  for (const double value : info.R)
    if (!std::isfinite(value))
      return false;
  for (const double value : info.P)
    if (!std::isfinite(value))
      return false;
  return true;
}

}  // namespace

void MapROS::setMap(SDFMap2D* map)
{
  this->map_ = map;
}

ros::Time MapROS::mapOutputStamp() const
{
  // Before the first complete source frame, periodic empty snapshots are
  // explicitly zero-stamped rather than claiming receipt-time freshness.
  return last_map_source_stamp_;
}

ros::Time MapROS::esdfOutputStamp() const
{
  // ESDF is computed asynchronously. Its timestamp must describe the buffer
  // actually finished by updateESDFMap, never a newer occupancy mutation.
  return last_esdf_source_stamp_;
}

void MapROS::resetSourceEpoch(bool clear_map_contents)
{
  mapping_history_.clear();
  last_map_source_stamp_ = ros::Time(0);
  pending_esdf_source_stamp_ = ros::Time(0);
  last_esdf_source_stamp_ = ros::Time(0);
  current_depth_source_stamp_ = ros::Time(0);
  last_semantic_stamp_ = ros::Time(0);
  last_camera_info_source_stamp_ = ros::Time(0);
  last_clock_now_ = ros::Time(0);
  local_updated_ = false;
  esdf_need_update_ = false;
  continue_over_depth_count_ = -1;
  itm_score_ = -1.0;
  semantic_target_.clear();
  if (depth_cloud_)
    depth_cloud_->clear();
  if (max_range_cloud_)
    max_range_cloud_->clear();
  if (filtered_depth_cloud2d_)
    filtered_depth_cloud2d_->clear();
  if (free_ray_cloud2d_)
    free_ray_cloud2d_->clear();
  if (clear_map_contents && map_)
    map_->resetForSourceEpoch();
}

void MapROS::observeClock(const ros::Time& now)
{
  if (!last_clock_now_.isZero() && now < last_clock_now_) {
    ROS_WARN("ROS clock reset detected; clearing all map, ESDF and semantic epoch state");
    resetSourceEpoch(true);
  }
  last_clock_now_ = now;
}

void MapROS::init()
{
  // Load camera intrinsic parameters from ROS parameter server
  node_.param("map_ros/fx", fx_, -1.0);
  node_.param("map_ros/fy", fy_, -1.0);
  node_.param("map_ros/cx", cx_, -1.0);
  node_.param("map_ros/cy", cy_, -1.0);
  node_.param("map_ros/use_camera_info", use_camera_info_, true);
  node_.param("map_ros/require_downward_camera", require_downward_camera_, false);
  node_.param("map_ros/mapping_history_sec", mapping_history_sec_, 5.0);
  node_.param("map_ros/semantic_match_tolerance", semantic_match_tolerance_, 0.05);
  node_.param("map_ros/camera_info_match_tolerance", camera_info_match_tolerance_, 0.01);
  // A frame contract cannot be checked without a CameraInfo frame. Even
  // legacy intrinsic settings therefore wait for one coherent CameraInfo.
  camera_info_ready_ = false;
  camera_width_ = camera_height_ = 0;
  camera_frame_id_.clear();
  last_camera_info_source_stamp_ = ros::Time(0);
  if (!std::isfinite(camera_info_match_tolerance_) || camera_info_match_tolerance_ < 0.0) {
    ROS_FATAL("map_ros/camera_info_match_tolerance must be finite and non-negative");
    throw std::invalid_argument("invalid CameraInfo/depth timestamp tolerance");
  }

  // Load depth filtering parameters
  node_.param("map_ros/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("map_ros/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("map_ros/self_filter_length", self_filter_length_, 0.70);
  node_.param("map_ros/self_filter_width", self_filter_width_, 0.70);
  node_.param("map_ros/self_filter_tolerance", self_filter_tolerance_, 1e-6);
  node_.param("map_ros/camera_forward_offset", camera_forward_offset_, 0.10);
  node_.param<std::string>(
      "map_ros/height_filter_reference", height_filter_reference_, "world");
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/filter_min_height", filter_min_height_, 0.5);
  node_.param("map_ros/filter_max_height", filter_max_height_, 0.88);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/depth_unit_scale", depth_unit_scale_, 0.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);
  node_.param("map_ros/frame_id", frame_id_, string("world"));
  node_.param("map_ros/virtual_ground_height", virtual_ground_height_, -0.28);
  node_.param("map_ros/max_source_age_sec", max_map_source_age_sec_, 1.0);
  node_.param("map_ros/max_future_sec", max_map_future_sec_, 0.05);

  if (!std::isfinite(mapping_history_sec_) || mapping_history_sec_ <= 0.0 ||
      !std::isfinite(semantic_match_tolerance_) || semantic_match_tolerance_ < 0.0 ||
      !std::isfinite(max_map_source_age_sec_) || max_map_source_age_sec_ <= 0.0 ||
      !std::isfinite(max_map_future_sec_) || max_map_future_sec_ < 0.0) {
    ROS_FATAL("Invalid map source/history timestamp policy");
    throw std::invalid_argument("invalid map source/history timestamp policy");
  }

  if (height_filter_reference_ != "world" && height_filter_reference_ != "sensor") {
    ROS_WARN("Unknown map_ros/height_filter_reference '%s'; using 'world'",
        height_filter_reference_.c_str());
    height_filter_reference_ = "world";
  }
  if (!(filter_min_height_ < filter_max_height_)) {
    ROS_FATAL("Invalid obstacle height band [%.3f, %.3f]",
        filter_min_height_, filter_max_height_);
    throw std::invalid_argument("map_ros obstacle height band is empty");
  }
  if (height_filter_reference_ == "sensor" &&
      !(filter_min_height_ < 0.0 && filter_max_height_ > 0.0)) {
    ROS_FATAL("Sensor-relative obstacle height offsets must straddle zero; got [%.3f, %.3f]",
        filter_min_height_, filter_max_height_);
    throw std::invalid_argument("sensor-relative obstacle height band must straddle zero");
  }
  ROS_INFO("Obstacle height filter uses %s reference with band [%.3f, %.3f] m",
      height_filter_reference_.c_str(), filter_min_height_, filter_max_height_);
  if (!(self_filter_length_ > 0.0) || !(self_filter_width_ > 0.0) ||
      self_filter_tolerance_ < 0.0) {
    ROS_FATAL("Invalid self footprint %.3f x %.3f m (tolerance %.6f)",
        self_filter_length_, self_filter_width_, self_filter_tolerance_);
    throw std::invalid_argument("map_ros self footprint is invalid");
  }
  ROS_INFO("Depth self-filter uses yaw-aligned %.2f x %.2f m footprint (tolerance %.1e)",
      self_filter_length_, self_filter_width_, self_filter_tolerance_);

  // Handle Habitat simulator vs real-world configuration
  bool is_real_world;
  node_.param("is_real_world", is_real_world, false);

  if (!is_real_world) {
    // Override depth parameters with Habitat simulator settings
    double habitat_max_depth, habitat_min_depth;
    node_.param("/habitat/simulator/agents/main_agent/sim_sensors/depth_sensor/max_depth",
        habitat_max_depth, -1.0);
    node_.param("/habitat/simulator/agents/main_agent/sim_sensors/depth_sensor/min_depth",
        habitat_min_depth, -1.0);
    if (habitat_max_depth != -1.0 && habitat_min_depth != -1.0) {
      depth_filter_maxdist_ = habitat_max_depth;
      depth_filter_mindist_ = habitat_min_depth;
      ROS_WARN("Using habitat simulator params, set depth_filter_range = [%.2f, %.2f] m",
          habitat_min_depth, habitat_max_depth);
    }
  }

  // Initialize point cloud data structures
  depth_cloud_.reset(new PointCloud3D());
  max_range_cloud_.reset(new PointCloud3D());
  filtered_depth_cloud2d_.reset(new PointCloud2D());
  free_ray_cloud2d_.reset(new PointCloud2D());

  // Image-dependent buffers are allocated from the received dimensions.
  proj_points_cnt_ = 0;
  depth_image_.reset(new cv::Mat);

  // Initialize state flags
  local_updated_ = false;
  esdf_need_update_ = false;
  mapping_enabled_ = false;
  navigation_enabled_ = false;
  last_map_source_stamp_ = ros::Time(0);
  pending_esdf_source_stamp_ = ros::Time(0);
  last_esdf_source_stamp_ = ros::Time(0);
  current_depth_source_stamp_ = ros::Time(0);
  last_clock_now_ = ros::Time(0);

  // Setup periodic timers for map updates and visualization
  esdf_timer_ = node_.createTimer(ros::Duration(0.1), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.25), &MapROS::visCallback, this);

  // Setup publishers for map visualization
  occupied_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupied", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/unknown", 10);
  free_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/free", 10);
  occupied_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupied_inflate", 10);

  object_grid_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupancy_object", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/grid_map/update_range", 10);
  depth_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/depth_cloud", 10);
  filtered_depth_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/filtered_depth_cloud", 10);
  filtered_object_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/filtered_object_cloud", 10);
  all_object_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/all_object_cloud", 10);
  over_depth_object_cloud_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/over_depth_object_cloud", 10);
  value_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/value_map", 10);
  confidence_map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/confidence_map", 10);
  map_commit_pub_ = node_.advertise<std_msgs::Header>("/grid_map/commit", 10);

  // Atomic semantic observations are the real/Gazebo interface. Legacy
  // subscribers remain optional for Habitat compatibility only.
  semantic_observation_sub_ = node_.subscribe(
      "/apexnav/vlm/semantic_observation", 10, &MapROS::semanticObservationCallback, this);
  camera_info_sub_ = node_.subscribe(
      "/map_ros/camera_info", 2, &MapROS::cameraInfoCallback, this);
  mapping_enabled_sub_ = node_.subscribe(
      "/apexnav/mission/mapping_enabled", 2, &MapROS::mappingEnabledCallback, this);
  navigation_enabled_sub_ = node_.subscribe(
      "/apexnav/mission/navigation_enabled", 2, &MapROS::navigationEnabledCallback, this);
  bool enable_legacy_semantics;
  node_.param("map_ros/enable_legacy_semantics", enable_legacy_semantics, false);
  if (enable_legacy_semantics) {
    detected_object_cloud_sub_ = node_.subscribe(
        "/detector/clouds_with_scores", 10, &MapROS::detectedObjectCloudCallback, this);
    itm_score_sub_ =
        node_.subscribe("/clip/cosine_score", 10, &MapROS::itmScoreCallback, this);
  }

  // Setup synchronized subscribers for depth image and pose data
  depth_sub_.reset(
      new message_filters::Subscriber<sensor_msgs::Image>(node_, "/map_ros/depth", 20));
  pose_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/map_ros/pose", 20));

  sync_image_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyImagePose>(
      MapROS::SyncPolicyImagePose(20), *depth_sub_, *pose_sub_));
  sync_image_pose_->setMaxIntervalDuration(ros::Duration(0.01));  // Set maximum temporal offset
  sync_image_pose_->registerCallback(boost::bind(&MapROS::depthPoseCallback, this, _1, _2));

  // Initialize object tracking variables
  continue_over_depth_count_ = -1;
  itm_score_ = -1.0;
  map_start_time_ = ros::Time::now();
}

void MapROS::itmScoreCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (!navigation_enabled_)
    return;
  itm_score_ = msg->data;
}

void MapROS::mappingEnabledCallback(const std_msgs::BoolConstPtr& msg)
{
  const bool enabled = msg->data;
  if (enabled == mapping_enabled_)
    return;

  mapping_enabled_ = enabled;
  resetSourceEpoch(mappingTransitionRequiresFullReset(enabled));
  if (enabled) {
    map_start_time_ = ros::Time::now();
    last_clock_now_ = map_start_time_;
  }
  ROS_WARN("Depth mapping %s", enabled ? "enabled" : "disabled");
}

void MapROS::navigationEnabledCallback(const std_msgs::BoolConstPtr& msg)
{
  const bool enabled = msg->data;
  if (enabled == navigation_enabled_)
    return;

  navigation_enabled_ = enabled;
  last_semantic_stamp_ = ros::Time(0);
  semantic_target_.clear();
  ROS_WARN("Semantic map updates %s", enabled ? "enabled" : "disabled");
}

void MapROS::visCallback(const ros::TimerEvent& e)
{
  vis_timer_.stop();
  observeClock(ros::Time::now());

  // Publish all visualization topics
  publishOccupied();
  publishInfOccupied();
  publishObjectMap();
  publishUnknown();
  publishFree();
  publishValueMap();
  // publishConfidenceMap();
  publishESDFMap();
  // publishUpdateRange();

  vis_timer_.start();
}

void MapROS::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  const ros::Time now = ros::Time::now();
  observeClock(now);
  if (!isAcceptableMapSourceStamp(msg->header.stamp.toSec(), now.toSec(),
          last_camera_info_source_stamp_.toSec(), max_map_source_age_sec_,
          max_map_future_sec_)) {
    ROS_WARN_THROTTLE(2.0,
        "Rejecting CameraInfo source stamp %.9f (zero, stale, future, or non-monotonic)",
        msg->header.stamp.toSec());
    return;
  }
  if (msg->header.frame_id.empty() || msg->width == 0 || msg->height == 0 ||
      !isFiniteCameraCalibration(*msg) || msg->K[0] <= 0.0 || msg->K[4] <= 0.0 ||
      msg->K[8] == 0.0 || msg->K[2] < 0.0 || msg->K[5] < 0.0 ||
      msg->K[2] >= static_cast<double>(msg->width) ||
      msg->K[5] >= static_cast<double>(msg->height)) {
    ROS_ERROR_THROTTLE(2.0, "Rejecting invalid CameraInfo or empty optical frame");
    return;
  }
  if (camera_info_ready_ &&
      (camera_width_ != static_cast<int>(msg->width) ||
          camera_height_ != static_cast<int>(msg->height))) {
    ROS_ERROR_THROTTLE(2.0, "CameraInfo dimensions changed during mapping; rejecting update");
    return;
  }
  if (camera_info_ready_ && (std::abs(fx_ - msg->K[0]) > 1e-6 ||
                                std::abs(fy_ - msg->K[4]) > 1e-6 ||
                                std::abs(cx_ - msg->K[2]) > 1e-6 ||
                                std::abs(cy_ - msg->K[5]) > 1e-6)) {
    ROS_ERROR_THROTTLE(2.0, "CameraInfo intrinsics changed during mapping; restart required");
    return;
  }
  if (camera_info_ready_ && camera_frame_id_ != msg->header.frame_id) {
    ROS_ERROR_THROTTLE(2.0, "CameraInfo frame changed from '%s' to '%s'; rejecting update",
        camera_frame_id_.c_str(), msg->header.frame_id.c_str());
    return;
  }
  fx_ = msg->K[0];
  fy_ = msg->K[4];
  cx_ = msg->K[2];
  cy_ = msg->K[5];
  camera_width_ = msg->width;
  camera_height_ = msg->height;
  camera_frame_id_ = msg->header.frame_id;
  last_camera_info_source_stamp_ = msg->header.stamp;
  camera_info_ready_ = true;
}

void MapROS::semanticObservationCallback(const plan_env::SemanticObservationConstPtr& msg)
{
  const ros::Time now = ros::Time::now();
  observeClock(now);
  if (!navigation_enabled_)
    return;
  if (msg->header.stamp.isZero() || msg->target_label.empty()) {
    ROS_WARN_THROTTLE(2.0, "Rejecting semantic observation without source stamp/target");
    return;
  }
  if (msg->header.frame_id != frame_id_) {
    ROS_WARN_THROTTLE(2.0, "Rejecting semantic observation in frame '%s' (expected '%s')",
        msg->header.frame_id.c_str(), frame_id_.c_str());
    return;
  }
  if (!(msg->confidence_scores.size() == msg->point_clouds.size() &&
          msg->confidence_scores.size() == msg->label_indices.size())) {
    ROS_ERROR("Rejecting inconsistent SemanticObservation arrays");
    return;
  }
  if (!msg->yolo_valid && !msg->point_clouds.empty()) {
    ROS_ERROR("Rejecting SemanticObservation with detections while yolo_valid is false");
    return;
  }
  if (msg->clip_valid && !std::isfinite(msg->clip_score)) {
    ROS_ERROR("Rejecting SemanticObservation with non-finite valid CLIP score");
    return;
  }
  const double semantic_max_age = std::max(max_map_source_age_sec_, mapping_history_sec_);
  if (!isAcceptableMapSourceStamp(msg->header.stamp.toSec(), now.toSec(),
          last_semantic_stamp_.toSec(), semantic_max_age, max_map_future_sec_)) {
    ROS_WARN_THROTTLE(2.0,
        "Rejecting semantic source stamp %.9f (zero, stale, future, or non-monotonic)",
        msg->header.stamp.toSec());
    return;
  }
  for (size_t index = 0; index < msg->point_clouds.size(); ++index) {
    const auto& cloud = msg->point_clouds[index];
    if (!isValidSemanticDetectionMetadata(
            msg->label_indices[index], msg->confidence_scores[index]) ||
        !isValidNestedCloudContract(msg->header.stamp.toSec(), msg->header.frame_id,
            cloud.header.stamp.toSec(), cloud.header.frame_id) ||
        !isFiniteXYZ32PointCloud(cloud)) {
      ROS_ERROR("Rejecting entire SemanticObservation: detection %zu violates metadata/cloud contract",
          index);
      return;
    }
  }

  auto best = mapping_history_.end();
  double best_delta = semantic_match_tolerance_ + 1.0;
  for (auto it = mapping_history_.begin(); it != mapping_history_.end(); ++it) {
    const double delta = std::abs((it->stamp - msg->header.stamp).toSec());
    if (delta < best_delta) {
      best_delta = delta;
      best = it;
    }
  }
  if (best == mapping_history_.end() || best_delta > semantic_match_tolerance_) {
    ROS_WARN_THROTTLE(2.0, "Dropping stale semantic observation; nearest map frame delta %.3f s",
        best_delta);
    return;
  }

  // Only after the complete observation and historical match pass validation
  // may any semantic state be committed.
  semantic_target_ = msg->target_label;

  // Run the existing object-map logic against the historical source frame,
  // never against whichever depth callback happened to run most recently.
  const Eigen::Vector3d current_camera_pos = camera_pos_;
  const Eigen::Quaterniond current_camera_q = camera_q_;
  PointCloud3D::Ptr current_depth_cloud(new PointCloud3D(*depth_cloud_));
  camera_pos_ = best->camera_pos;
  camera_q_ = best->camera_q;
  depth_cloud_.reset(new PointCloud3D(*best->depth_cloud));
  itm_score_ = msg->clip_valid && std::isfinite(msg->clip_score) ? msg->clip_score : -1.0;
  plan_env::MultipleMasksWithConfidencePtr legacy(new plan_env::MultipleMasksWithConfidence());
  legacy->point_clouds = msg->point_clouds;
  legacy->confidence_scores = msg->confidence_scores;
  legacy->label_indices = msg->label_indices;
  if (msg->yolo_valid)
    processDetectedObjectCloud(legacy, best->stamp);
  if (msg->clip_valid && std::isfinite(msg->clip_score))
    map_->value_map_->updateValueMap(Eigen::Vector2d(best->camera_pos.x(), best->camera_pos.y()),
        best->camera_yaw, best->free_grids, msg->clip_score);
  last_semantic_stamp_ = msg->header.stamp;
  camera_pos_ = current_camera_pos;
  camera_q_ = current_camera_q;
  depth_cloud_ = current_depth_cloud;
}

void MapROS::detectedObjectCloudCallback(const plan_env::MultipleMasksWithConfidenceConstPtr& msg)
{
  (void)msg;
  // This legacy message has no header. Receipt time must never be converted
  // into a map source time, so only the atomic SemanticObservation path can
  // feed semantic object mapping.
  ROS_WARN_THROTTLE(2.0,
      "Dropping headerless legacy object cloud; use SemanticObservation with a source stamp");
}

void MapROS::processDetectedObjectCloud(
    const plan_env::MultipleMasksWithConfidenceConstPtr& msg, const ros::Time& source_stamp)
{
  if (!navigation_enabled_)
    return;
  if (source_stamp.isZero()) {
    ROS_WARN_THROTTLE(2.0, "Dropping semantic object cloud without matched depth source stamp");
    return;
  }
  // Validate message structure consistency
  if (!(msg->confidence_scores.size() == msg->point_clouds.size() &&
          msg->confidence_scores.size() == msg->label_indices.size())) {
    ROS_ERROR("[Bug] The MultipleMasksWithConfidence msg is wrong!!!");
    return;
  }
  // Defense in depth: do not mutate over-depth/object state until every
  // detection has passed the fixed-size class, finite and PointCloud2 layout
  // contracts. The atomic callback performs the same preflight validation.
  for (size_t index = 0; index < msg->confidence_scores.size(); ++index) {
    const auto& cloud = msg->point_clouds[index];
    if (!isValidSemanticDetectionMetadata(
            msg->label_indices[index], msg->confidence_scores[index]) ||
        !isValidNestedCloudContract(source_stamp.toSec(), frame_id_,
            cloud.header.stamp.toSec(), cloud.header.frame_id) ||
        !isFiniteXYZ32PointCloud(cloud)) {
      ROS_ERROR("Rejecting entire semantic object frame at detection %zu", index);
      return;
    }
  }

  auto t1 = ros::Time::now();

  // Optical +Z is the view direction. Avoid Euler-angle tests on an optical
  // frame, which classify a horizontal camera as downward after axis rotation.
  const bool camera_looking_down = camera_q_.toRotationMatrix().col(2).z() < -0.95;
  if (require_downward_camera_ && !camera_looking_down)
    return;

  // Backup previous over-depth object cloud for consistency tracking
  auto last_over_depth_cloud =
      std::make_shared<PointCloud3D>(*map_->object_map2d_->over_depth_object_cloud_);
  map_->object_map2d_->over_depth_object_cloud_.reset(new PointCloud3D());

  // Initialize point cloud processing tools and containers
  pcl::VoxelGrid<Point3D> voxel_filter;
  PointCloud3D::Ptr all_object_cloud(new PointCloud3D());
  PointCloud3D::Ptr filtered_all_object_cloud(new PointCloud3D());
  vector<DetectedObject> detected_objects;

  // Process each detected object in the message
  for (int i = 0; i < (int)msg->confidence_scores.size(); i++) {
    auto cloud = msg->point_clouds[i];
    auto confidence_score = msg->confidence_scores[i];
    auto label = msg->label_indices[i];

    // Convert ROS message to PCL point cloud
    PointCloud3D::Ptr single_object_cloud(new PointCloud3D());
    pcl::fromROSMsg(cloud, *single_object_cloud);
    *all_object_cloud += *single_object_cloud;

    // Apply voxel grid downsampling to reduce computational load
    voxel_filter.setInputCloud(single_object_cloud);
    voxel_filter.setLeafSize(0.04f, 0.04f, 0.06f);
    voxel_filter.filter(*single_object_cloud);

    // Filter out points beyond sensor accuracy range (>5m depth is unreliable)
    PointCloud3D::Ptr tmp_object_cloud(new PointCloud3D());
    PointCloud3D::Ptr over_depth_object_cloud(new PointCloud3D());
    for (auto object_pt : single_object_cloud->points) {
      Eigen::Vector3d object_pt3d = Eigen::Vector3d(object_pt.x, object_pt.y, object_pt.z);
      if ((object_pt3d - camera_pos_).norm() > depth_filter_maxdist_ - 0.10) {
        // Store over-depth points for target objects (label == 0) for tracking consistency
        if (label == 0)
          over_depth_object_cloud->points.push_back(object_pt);
        continue;
      }
      tmp_object_cloud->points.push_back(object_pt);
    }
    single_object_cloud = tmp_object_cloud;

    // Skip objects that are entirely beyond valid depth range
    if (single_object_cloud->points.empty()) {
      if (!over_depth_object_cloud->points.empty()) {
        ROS_WARN_THROTTLE(2.0,
            "Semantic object cloud is entirely beyond the valid %.2fm depth range; "
            "retaining it only for over-depth consistency tracking.", depth_filter_maxdist_);
        *map_->object_map2d_->over_depth_object_cloud_ += *over_depth_object_cloud;
      }
      continue;
    }

    // Apply DBSCAN clustering to remove noise and outliers
    single_object_cloud = dbscan(single_object_cloud, 0.12f, 10);
    if (single_object_cloud == nullptr) {
      ROS_ERROR("After DBSCAN, no point cloud cluster!!");
      continue;
    }

    if (single_object_cloud->points.empty()) {
      ROS_ERROR("Single object point cloud is empty!!!");
      continue;
    }

    // Accumulate filtered object data
    *filtered_all_object_cloud += *single_object_cloud;
    DetectedObject detected_object;
    detected_object.cloud = single_object_cloud;
    detected_object.score = confidence_score;
    detected_object.label = label;
    detected_object.source_stamp = source_stamp;
    detected_objects.push_back(detected_object);
  }

  // Maintain consistency in over-depth object tracking
  if (continue_over_depth_count_ == -1 &&
      !map_->object_map2d_->over_depth_object_cloud_->points.empty())
    continue_over_depth_count_ = 0;
  else if (continue_over_depth_count_ <= 4 && continue_over_depth_count_ >= 0) {
    continue_over_depth_count_++;
    *map_->object_map2d_->over_depth_object_cloud_ = *last_over_depth_cloud;
  }
  else {
    continue_over_depth_count_ = -1;
  }

  // Publish visualization point clouds for debugging and monitoring
  publishPointCloud(filtered_object_cloud_pub_, filtered_all_object_cloud, source_stamp);
  publishPointCloud(all_object_cloud_pub_, all_object_cloud, source_stamp);
  publishPointCloud(
      over_depth_object_cloud_pub_, map_->object_map2d_->over_depth_object_cloud_, source_stamp);

  // Update object map with processed detection results
  *map_->object_map2d_->all_object_clouds_ = *filtered_all_object_cloud;
  vector<int> detected_object_cluster_ids;
  map_->inputObjectCloud2D(detected_objects, detected_object_cluster_ids);

  // Optional: Log detected object IDs for debugging
  // for (auto object_id : detected_object_cluster_ids)
  //   ROS_INFO("Detected object id is %d", object_id);

  // Extract observation data from depth sensor for objects not detected by vision
  getObservationObjectsCloud(detected_object_cluster_ids);

  double object_map_process_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(
      10.0, "[Calculating Time] Object Map process time = %.3f s", object_map_process_time);
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/)
{
  observeClock(ros::Time::now());
  if (!mapping_enabled_ || !esdf_need_update_)
    return;

  esdf_timer_.stop();

  auto t1 = ros::Time::now();
  const ros::Time source_stamp = pending_esdf_source_stamp_;
  map_->updateESDFMap();
  if (!source_stamp.isZero())
    last_esdf_source_stamp_ = source_stamp;
  pending_esdf_source_stamp_ = ros::Time(0);
  esdf_need_update_ = false;
  double esdf_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] ESDF Map process time = %.3f s", esdf_time);

  esdf_timer_.start();
}

void MapROS::depthPoseCallback(
    const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose)
{
  if (!mapping_enabled_)
    return;
  const ros::Time now = ros::Time::now();
  observeClock(now);
  if (!camera_info_ready_) {
    ROS_WARN_THROTTLE(2.0, "Waiting for valid CameraInfo before mapping");
    return;
  }
  if (img->header.stamp.isZero() || pose->header.stamp.isZero() ||
      std::abs((img->header.stamp - pose->header.stamp).toSec()) > 0.01) {
    ROS_WARN_THROTTLE(2.0, "Rejecting depth/pose without coherent source timestamps");
    return;
  }
  if (!isValidMapFrameContract(frame_id_, pose->header.frame_id, pose->child_frame_id,
          img->header.frame_id, camera_frame_id_)) {
    ROS_ERROR_THROTTLE(2.0,
        "Rejecting frame contract: map='%s' pose='%s' child='%s' depth='%s' camera_info='%s'",
        frame_id_.c_str(), pose->header.frame_id.c_str(), pose->child_frame_id.c_str(),
        img->header.frame_id.c_str(), camera_frame_id_.c_str());
    return;
  }
  const auto& pose_position = pose->pose.pose.position;
  const auto& pose_orientation = pose->pose.pose.orientation;
  if (!isFinitePoseAndReasonableQuaternion(pose_position.x, pose_position.y, pose_position.z,
          pose_orientation.x, pose_orientation.y, pose_orientation.z, pose_orientation.w)) {
    ROS_ERROR_THROTTLE(2.0, "Rejecting non-finite pose or unreasonable pose quaternion");
    return;
  }
  const ros::Time source_stamp = img->header.stamp;
  if (!isAcceptableMapSourceStamp(source_stamp.toSec(), now.toSec(),
          last_map_source_stamp_.toSec(), max_map_source_age_sec_, max_map_future_sec_)) {
    ROS_WARN_THROTTLE(2.0,
        "Rejecting depth source stamp %.9f (zero, stale, future, or non-monotonic)",
        source_stamp.toSec());
    return;
  }
  if (!isSourceStampCoherent(source_stamp.toSec(),
          last_camera_info_source_stamp_.toSec(), camera_info_match_tolerance_)) {
    ROS_WARN_THROTTLE(2.0,
        "Rejecting depth source %.9f without coherent CameraInfo source (last %.9f, tolerance %.3fs)",
        source_stamp.toSec(), last_camera_info_source_stamp_.toSec(),
        camera_info_match_tolerance_);
    return;
  }
  if (camera_width_ > 0 && (camera_width_ != static_cast<int>(img->width) ||
                              camera_height_ != static_cast<int>(img->height))) {
    ROS_ERROR_THROTTLE(2.0, "Depth dimensions differ from CameraInfo");
    return;
  }
  // Extract camera pose from odometry message
  camera_pos_(0) = pose_position.x;
  camera_pos_(1) = pose_position.y;
  camera_pos_(2) = pose_position.z;
  camera_q_ = Eigen::Quaterniond(pose_orientation.w, pose_orientation.x, pose_orientation.y,
      pose_orientation.z);
  camera_q_.normalize();

  // ROS optical +Z is the viewing direction. Project that axis into the map
  // plane; Euler-Z of an optical quaternion is offset by roughly 90 degrees.
  const Eigen::Vector3d optical_forward = camera_q_.toRotationMatrix().col(2);
  const double camera_yaw = std::atan2(optical_forward.y(), optical_forward.x());
  Eigen::Vector2d camera_pos = Eigen::Vector2d(camera_pos_(0), camera_pos_(1));

  // Skip processing if camera is outside map bounds
  if (!map_->isInMap(camera_pos))
    return;

  // Convert every supported encoding to CV_32FC1 metres. Unsupported encodings
  // are rejected instead of guessed.
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(img, img->encoding);
  }
  catch (const cv_bridge::Exception& e) {
    ROS_ERROR_THROTTLE(2.0, "Depth conversion failed: %s", e.what());
    return;
  }
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    const double scale = depth_unit_scale_ > 0.0 ? depth_unit_scale_ :
        (depth_filter_maxdist_ - depth_filter_mindist_);
    cv_ptr->image.convertTo(*depth_image_, CV_32FC1, scale,
        depth_unit_scale_ > 0.0 ? 0.0 : depth_filter_mindist_);
  }
  else if (img->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    if (depth_unit_scale_ <= 0.0) {
      ROS_ERROR_THROTTLE(2.0, "16UC1 requires positive map_ros/depth_unit_scale");
      return;
    }
    cv_ptr->image.convertTo(*depth_image_, CV_32FC1, depth_unit_scale_);
  }
  else if (img->encoding == sensor_msgs::image_encodings::TYPE_8UC1 && depth_unit_scale_ <= 0.0) {
    cv_ptr->image.convertTo(*depth_image_, CV_32FC1,
        (depth_filter_maxdist_ - depth_filter_mindist_) / 255.0, depth_filter_mindist_);
  }
  else {
    ROS_ERROR_THROTTLE(2.0, "Unsupported depth encoding '%s'", img->encoding.c_str());
    return;
  }
  current_depth_source_stamp_ = source_stamp;

  auto t1 = ros::Time::now();

  // Process depth image into 3D point cloud and filter to 2D representation
  processDepthImage();
  filterPointCloudToXY();

  // Update occupancy grid with filtered depth data.
  vector<Eigen::Vector2i> free_grids;
  const bool has_map_observation = !filtered_depth_cloud2d_->empty() || !free_ray_cloud2d_->empty();
  if (!has_map_observation) {
    ROS_WARN_THROTTLE(2.0, "Depth frame had no valid mapping endpoints; source stamp not committed");
    return;
  }
  map_->inputDepthCloud2D(
      filtered_depth_cloud2d_, free_ray_cloud2d_, camera_pos_, free_grids);
  // inputDepthCloud2D creates free_grids; dilating before it is a no-op.
  dilateGrids(free_grids, 1);
  // A body-mounted depth camera can see the airframe/propeller plane. Those
  // body-fixed returns must not accumulate into a world-fixed obstacle ring.
  // Exclude current vehicle volume and clear any historic self returns before
  // inflation/ESDF updates.
  const SelfFilterFootprint self_footprint{
      self_filter_length_, self_filter_width_, self_filter_tolerance_};
  map_->clearOccupancyFootprint(currentVehicleCenter2D(), currentVehicleYaw(), self_footprint);
  double process_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] Grid Map process time = %.3f s", process_time);

  // Cache the exact frame needed when asynchronous VLM results return. The
  // occupancy map never waits for semantic inference.
  MappingFrame frame;
  frame.stamp = img->header.stamp;
  frame.camera_pos = camera_pos_;
  frame.camera_q = camera_q_;
  frame.camera_yaw = camera_yaw;
  frame.free_grids = free_grids;
  frame.depth_cloud.reset(new PointCloud3D(*depth_cloud_));
  mapping_history_.push_back(frame);
  ros::Time cutoff;
  cutoff.fromSec(clampedHistoryCutoffSec(frame.stamp.toSec(), mapping_history_sec_));
  while (!mapping_history_.empty() && mapping_history_.front().stamp < cutoff)
    mapping_history_.pop_front();

  // Trigger ESDF update if local map has been updated
  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    pending_esdf_source_stamp_ = source_stamp;
    esdf_need_update_ = true;
    local_updated_ = false;
  }
  // Do not let receipt/timer time make an old frame appear fresh. Commit only
  // after all occupancy, self-history cleanup and inflation work has succeeded.
  last_map_source_stamp_ = source_stamp;
  std_msgs::Header commit;
  commit.stamp = source_stamp;
  commit.frame_id = frame_id_;
  map_commit_pub_.publish(commit);
}

void MapROS::processDepthImage()
{
  proj_points_cnt_ = 0;
  int cols = depth_image_->cols;
  int rows = depth_image_->rows;
  double depth;
  Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
  Eigen::Vector3d pt_cur, pt_world;
  depth_cloud_->clear();
  max_range_cloud_->clear();
  const int estimated_points =
      std::max(1, (rows / std::max(1, skip_pixel_)) * (cols / std::max(1, skip_pixel_)));
  depth_cloud_->points.reserve(estimated_points);
  max_range_cloud_->points.reserve(estimated_points);

  // Iterate through depth image pixels with margin and skipping for efficiency
  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {
      depth = depth_image_->at<float>(v, u);

      // Apply depth range filtering
      if (std::isnan(depth) || depth <= 0.0 || depth < depth_filter_mindist_)
        continue;
      // A return at or beyond the mapping limit means that this ray did not
      // hit an obstacle inside the local map. Keep it as a separate free ray.
      // Clamping it into the obstacle cloud creates a false circular wall at
      // depth_filter_maxdist_ during a yaw scan.
      const bool max_range_ray = !std::isfinite(depth) || depth >= depth_filter_maxdist_;
      if (max_range_ray)
        depth = depth_filter_maxdist_;

      // Project pixel to 3D camera coordinates
      pt_cur(0) = (u - cx_) * depth / fx_;
      pt_cur(1) = (v - cy_) * depth / fy_;
      pt_cur(2) = depth;

      // Transform to world coordinates
      pt_world = camera_r * pt_cur + camera_pos_;
      if (!pt_world.allFinite())
        continue;
      Point3D pt;
      pt.x = pt_world[0];
      pt.y = pt_world[1];
      pt.z = pt_world[2];
      if (max_range_ray)
        max_range_cloud_->points.push_back(pt);
      else
        depth_cloud_->points.push_back(pt);
      ++proj_points_cnt_;
    }
  }
  depth_cloud_->width = depth_cloud_->points.size();
  depth_cloud_->height = 1;
  depth_cloud_->is_dense = false;
  max_range_cloud_->width = max_range_cloud_->points.size();
  max_range_cloud_->height = 1;
  max_range_cloud_->is_dense = false;
  publishPointCloud(depth_cloud_pub_, depth_cloud_, current_depth_source_stamp_);
}

/**
 * @brief Extract undetected objects from depth observation data
 *
 * Identifies objects that appear in depth sensor data but weren't detected by
 * the vision system. Uses bounding box filtering to separate already detected
 * objects from potential undetected ones. Assigns zero confidence to undetected objects.
 *
 * @param filter_object_ids List of already detected object cluster IDs to filter out
 */
void MapROS::getObservationObjectsCloud(const std::vector<int>& filter_object_ids)
{
  // Downsample depth cloud for efficient processing
  PointCloud3D::Ptr filtered_depth_cloud(new PointCloud3D());
  pcl::VoxelGrid<Point3D> voxel_filter;
  voxel_filter.setInputCloud(depth_cloud_);
  voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_filter.filter(*filtered_depth_cloud);

  // Get object bounding boxes and create filter flags
  vector<Vector3d> bmins, bmaxs;
  map_->object_map2d_->getObjectBoxes(bmins, bmaxs);
  vector<char> filter_object_flag(bmins.size(), 0);
  for (auto filter_object_id : filter_object_ids)
    if (filter_object_id >= 0 && filter_object_id < static_cast<int>(filter_object_flag.size()))
      filter_object_flag[filter_object_id] = 1;

  // Use CropBox filter to extract points within object bounding boxes
  pcl::CropBox<Point3D> crop_box_filter;
  crop_box_filter.setInputCloud(filtered_depth_cloud);
  vector<pcl::shared_ptr<PointCloud3D>> observation_clouds;

  for (int i = 0; i < (int)bmins.size(); i++) {
    PointCloud3D::Ptr cloud_filtered(new PointCloud3D);
    if (filter_object_flag[i])
      observation_clouds.push_back(cloud_filtered);  // Empty cloud for detected objects
    else {
      // Extract points within bounding box for undetected objects
      double inf = 0.2f;  // Inflation factor for bounding box
      Eigen::Vector4f min_point(bmins[i][0] - inf, bmins[i][1] - inf, bmins[i][2] - inf, 1.0);
      Eigen::Vector4f max_point(bmaxs[i][0] + inf, bmaxs[i][1] + inf, bmaxs[i][2] + inf, 1.0);
      crop_box_filter.setMin(min_point);
      crop_box_filter.setMax(max_point);
      crop_box_filter.filter(*cloud_filtered);
      observation_clouds.push_back(cloud_filtered);
    }
  }

  // Update object map with observation data (using max of 0 and ITM score)
  map_->object_map2d_->inputObservationObjectsCloud(observation_clouds, max(0.0, itm_score_));
}

/**
 * @brief Filter and process 3D point cloud to 2D occupancy grid
 */
void MapROS::filterPointCloudToXY()
{
  // Default ground height assumption (currently set to 0)
  double cur_floor_height = 0.0;
  double virtual_ground = virtual_ground_height_;

  auto t1 = ros::Time::now();
  PointCloud3D::Ptr filtered_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr down_depth_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr down_max_range_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr under_ground_cloud_3d(new PointCloud3D());
  PointCloud2D::Ptr under_ground_cloud_2d(new PointCloud2D());
  const Eigen::Vector2d vehicle_center = currentVehicleCenter2D();
  const auto obstacle_height_bounds = obstacleHeightBounds();
  const double obstacle_min_z = obstacle_height_bounds.first;
  const double obstacle_max_z = obstacle_height_bounds.second;
  int self_filtered_points = 0;
  int below_obstacle_band_points = 0;
  int above_obstacle_band_points = 0;

  // Downsample point cloud for efficient processing
  pcl::VoxelGrid<Point3D> voxel_filter;
  voxel_filter.setInputCloud(depth_cloud_);
  voxel_filter.setLeafSize(0.04f, 0.04f, 0.1f);  // Different resolution for XY vs Z
  voxel_filter.filter(*down_depth_cloud_3d);
  voxel_filter.setInputCloud(max_range_cloud_);
  voxel_filter.filter(*down_max_range_cloud_3d);

  filtered_depth_cloud2d_->clear();
  free_ray_cloud2d_->clear();

  // Separate points by height categories
  for (int i = 0; i < (int)down_depth_cloud_3d->points.size(); i++) {
    Point3D pt;
    pt.x = down_depth_cloud_3d->points[i].x;
    pt.y = down_depth_cloud_3d->points[i].y;
    pt.z = down_depth_cloud_3d->points[i].z;

    // Reject returns inside the vehicle volume before either obstacle or
    // virtual-ground processing. These are physically incapable of being a
    // static external obstacle and are normally airframe/self reflections.
    const SelfFilterFootprint self_footprint{
        self_filter_length_, self_filter_width_, self_filter_tolerance_};
    if (isInsideSelfFootprint(pt.x, pt.y, vehicle_center.x(), vehicle_center.y(),
            currentVehicleYaw(), self_footprint)) {
      ++self_filtered_points;
      continue;
    }

    // Points below virtual ground (for virtual ground generation)
    if (down_depth_cloud_3d->points[i].z < cur_floor_height + virtual_ground)
      under_ground_cloud_3d->points.push_back(pt);
    // Points in obstacle height range
    else if (down_depth_cloud_3d->points[i].z > obstacle_min_z &&
             down_depth_cloud_3d->points[i].z < obstacle_max_z)
      filtered_cloud_3d->points.push_back(pt);
    else if (down_depth_cloud_3d->points[i].z <= obstacle_min_z)
      ++below_obstacle_band_points;
    else
      ++above_obstacle_band_points;
  }

  pcl::RadiusOutlierRemoval<Point3D> outrem;

  // Remove outliers from obstacle points (handles noisy depth data from datasets)
  if (!filtered_cloud_3d->points.empty()) {
    outrem.setInputCloud(filtered_cloud_3d);
    outrem.setRadiusSearch(0.3);         // Search radius for neighbors
    outrem.setMinNeighborsInRadius(35);  // Minimum neighbor threshold
    outrem.filter(*filtered_cloud_3d);
  }

  publishPointCloud(filtered_depth_cloud_pub_, filtered_cloud_3d, current_depth_source_stamp_);
  ROS_INFO_THROTTLE(2.0,
      "Depth self-filter removed %d points inside yaw-aligned %.2f x %.2f m vehicle footprint",
      self_filtered_points, self_filter_length_, self_filter_width_);
  ROS_INFO_THROTTLE(2.0,
      "Depth obstacle height band %s [%.2f, %.2f]m: kept %zu, rejected %d low / %d high",
      height_filter_reference_.c_str(), obstacle_min_z, obstacle_max_z,
      filtered_cloud_3d->points.size(), below_obstacle_band_points,
      above_obstacle_band_points);

  // Project 3D obstacle points to 2D for occupancy mapping
  for (auto pt : filtered_cloud_3d->points) {
    Point2D pt_xy;
    pt_xy.x = pt.x;
    pt_xy.y = pt.y;
    filtered_depth_cloud2d_->points.push_back(pt_xy);
  }

  // Max-range samples clear visible free space but must never contribute an
  // occupied endpoint. Apply the same navigation-height and body filters used
  // for obstacle returns so these rays describe the same 2-D map slice.
  for (const auto& pt : down_max_range_cloud_3d->points) {
    const SelfFilterFootprint self_footprint{
        self_filter_length_, self_filter_width_, self_filter_tolerance_};
    if (isInsideSelfFootprint(pt.x, pt.y, vehicle_center.x(), vehicle_center.y(),
            currentVehicleYaw(), self_footprint))
      continue;
    if (pt.z <= obstacle_min_z || pt.z >= obstacle_max_z)
      continue;
    Point2D pt_xy;
    pt_xy.x = pt.x;
    pt_xy.y = pt.y;
    free_ray_cloud2d_->points.push_back(pt_xy);
  }
  ROS_INFO_THROTTLE(2.0, "Depth map endpoints: %zu occupied, %zu max-range free rays",
      filtered_depth_cloud2d_->points.size(), free_ray_cloud2d_->points.size());

  // Remove outliers from under-ground points (handles noisy depth data)
  if (!under_ground_cloud_3d->points.empty()) {
    outrem.setInputCloud(under_ground_cloud_3d);
    outrem.setRadiusSearch(0.21);        // Smaller search radius for ground points
    outrem.setMinNeighborsInRadius(40);  // Higher neighbor threshold
    outrem.filter(*under_ground_cloud_3d);
  }

  const bool camera_looking_down = camera_q_.toRotationMatrix().col(2).z() < -0.95;
  if (camera_looking_down && !under_ground_cloud_3d->points.empty()) {
    for (auto pt : under_ground_cloud_3d->points) {
      Eigen::Vector3d pt_pos = Eigen::Vector3d(pt.x, pt.y, pt.z);
      Eigen::Vector2d ground_pos;

      // Interpolate ray from camera to point, finding intersection with virtual ground
      if (interpolateLineAtZ(pt_pos, camera_pos_, cur_floor_height + virtual_ground, ground_pos)) {
        Point2D pt_xy;
        pt_xy.x = ground_pos(0);
        pt_xy.y = ground_pos(1);
        filtered_depth_cloud2d_->points.push_back(pt_xy);
        under_ground_cloud_2d->points.push_back(pt_xy);
      }
    }
    map_->inputVirtualGround(under_ground_cloud_2d);
  }

  double filter_time = (ros::Time::now() - t1).toSec();
  ROS_WARN_COND(filter_time > 0.1, "Filter point cloud time maybe a little long = %.3f ms",
      filter_time * 1000);
}

Eigen::Vector2d MapROS::currentVehicleCenter2D() const
{
  // Optical +Z is camera forward. The Gazebo camera is mounted 0.10 m in
  // front of base_link, so translate backwards to recover the body center.
  const Eigen::Vector3d optical_forward = camera_q_.toRotationMatrix().col(2);
  return camera_pos_.head(2) - camera_forward_offset_ * optical_forward.head(2);
}

double MapROS::currentVehicleYaw() const
{
  const Eigen::Vector3d optical_forward = camera_q_.toRotationMatrix().col(2);
  return std::atan2(optical_forward.y(), optical_forward.x());
}

std::pair<double, double> MapROS::obstacleHeightBounds() const
{
  // PX4/MAVROS local frames are not required to place the launch floor at
  // world z=0. In sensor-relative mode the 2-D occupancy slice follows the
  // vehicle, so surfaces below/above its swept vertical envelope cannot turn
  // into false horizontal walls when the local origin changes.
  const double reference_z =
      height_filter_reference_ == "sensor" ? camera_pos_.z() : 0.0;
  return std::make_pair(
      reference_z + filter_min_height_, reference_z + filter_max_height_);
}

bool MapROS::interpolateLineAtZ(
    const Eigen::Vector3d& A, const Eigen::Vector3d& B, double target_z, Eigen::Vector2d& P)
{
  // Check if target_z is between A.z and B.z (intersection possible)
  if ((A.z() - target_z) * (B.z() - target_z) > 0)
    return false;  // target_z not within segment bounds

  // Calculate interpolation parameter t (0 = point A, 1 = point B)
  double t = (target_z - A.z()) / (B.z() - A.z());

  // Linear interpolation for X and Y coordinates
  double x = A.x() + t * (B.x() - A.x());
  double y = A.y() + t * (B.y() - A.y());
  P = Eigen::Vector2d(x, y);
  return true;
}

/**
 * @brief DBSCAN clustering algorithm to extract largest point cloud cluster
 *
 * Applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
 * to identify and return the largest cluster from a point cloud. This is used
 * to filter noise and extract the main object point cloud, assuming each object
 * consists of a single dominant cluster.
 *
 * @param cloud Input point cloud to cluster
 * @param eps Maximum distance between points in the same cluster (neighborhood radius)
 * @param minPts Minimum number of points required to form a dense region (cluster)
 * @return Pointer to largest cluster point cloud, or nullptr if clustering fails
 */
PointCloud3D::Ptr MapROS::dbscan(const PointCloud3D::Ptr& cloud, double eps, int minPts)
{
  if (cloud->empty()) {
    ROS_ERROR("[DBSCAN] Input cloud is empty!");
    return nullptr;
  }

  // Build KD-tree for efficient neighbor search
  pcl::search::KdTree<Point3D>::Ptr tree(new pcl::search::KdTree<Point3D>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;

  // Use PCL's EuclideanClusterExtraction to implement DBSCAN-like clustering
  pcl::EuclideanClusterExtraction<Point3D> ec;
  ec.setClusterTolerance(eps);                 // Neighborhood radius
  ec.setMinClusterSize(minPts);                // Minimum points per cluster
  ec.setMaxClusterSize(cloud->points.size());  // Maximum cluster size (full cloud)
  ec.setSearchMethod(tree);                    // Set KD-Tree for neighbor search
  ec.setInputCloud(cloud);                     // Input point cloud
  ec.extract(cluster_indices);                 // Extract clustering results

  // Return null if no clusters found
  if (cluster_indices.empty()) {
    ROS_WARN("[DBSCAN] No clusters found!");
    return nullptr;
  }

  // Find the largest cluster by counting points
  int largest_cluster_index = -1;
  size_t max_size = 0;
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    if (cluster_indices[i].indices.size() > max_size) {
      max_size = cluster_indices[i].indices.size();
      largest_cluster_index = i;
    }
  }

  // Create new point cloud containing only the largest cluster
  PointCloud3D::Ptr largest_cluster(new PointCloud3D);
  for (int idx : cluster_indices[largest_cluster_index].indices)
    largest_cluster->points.push_back(cloud->points[idx]);
  return largest_cluster;
}
}  // namespace apexnav_planner
