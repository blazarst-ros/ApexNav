/**
 * @file map_ros.cpp
 * @brief Implementation of ROS interface for 2D SDF mapping system
 *
 * Multi-agent extension: Each agent has independent sensor state (pose, depth, ITM)
 * stored in AgentState. All agents contribute to shared SDFMap2D, ObjectMap2D, and
 * ValueMap2D with mutex-protected writes.
 *
 * @author Zager-Zhang
 */

#include <plan_env/map_ros.h>
#include <plan_env/semantic_observability.h>

#include <algorithm>
#include <cmath>

namespace apexnav_planner {

void MapROS::setMap(SDFMap2D* map)
{
  this->map_ = map;
}

void MapROS::init()
{
  // Load camera intrinsic parameters from ROS parameter server
  node_.param("map_ros/fx", fx_, -1.0);
  node_.param("map_ros/fy", fy_, -1.0);
  node_.param("map_ros/cx", cx_, -1.0);
  node_.param("map_ros/cy", cy_, -1.0);

  // Load depth filtering parameters
  node_.param("map_ros/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("map_ros/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/filter_min_height", filter_min_height_, 0.5);
  node_.param("map_ros/filter_max_height", filter_max_height_, 1.30);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);
  node_.param("map_ros/frame_id", frame_id_, string("world"));
  node_.param("map_ros/virtual_ground_height", virtual_ground_height_, -0.28);
  node_.param("map_ros/loose_semantic_evidence_debug", loose_semantic_evidence_debug_, true);

  // Handle Habitat simulator vs real-world configuration
  bool is_real_world;
  node_.param("is_real_world", is_real_world, false);

  if (!is_real_world) {
    // Override depth parameters with Habitat simulator settings
    double habitat_max_depth, habitat_min_depth;
    node_.param("/habitat/simulator/agents/agent_0/sim_sensors/depth_sensor/max_depth",
        habitat_max_depth, -1.0);
    node_.param("/habitat/simulator/agents/agent_0/sim_sensors/depth_sensor/min_depth",
        habitat_min_depth, -1.0);
    if (habitat_max_depth != -1.0 && habitat_min_depth != -1.0) {
      depth_filter_maxdist_ = habitat_max_depth;
      depth_filter_mindist_ = habitat_min_depth;
      ROS_WARN("Using habitat simulator params, set depth_filter_range = [%.2f, %.2f] m",
          habitat_min_depth, habitat_max_depth);
    }
  }

  // Initialize per-agent state
  agents_.resize(NUM_AGENTS_);
  for (int id = 0; id < NUM_AGENTS_; ++id) {
    agents_[id].camera_pos_.setZero();
    agents_[id].camera_q_.setIdentity();
    agents_[id].depth_cloud_.reset(new PointCloud3D());
    agents_[id].depth_cloud_->points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
    agents_[id].proj_points_cnt_ = 0;
    agents_[id].filtered_depth_cloud2d_.reset(new PointCloud2D());
    agents_[id].under_ground_cloud2d_.reset(new PointCloud2D());
    agents_[id].over_depth_object_cloud_.reset(new PointCloud3D());
    agents_[id].depth_image_.reset(new cv::Mat);
    agents_[id].continue_over_depth_count_ = -1;
    agents_[id].itm_score_ = -1.0;
  }

  // Initialize state flags (shared map state)
  local_updated_ = false;
  esdf_need_update_ = false;

  // Setup periodic timers for map updates and visualization (shared)
  esdf_timer_ = node_.createTimer(ros::Duration(0.1), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.25), &MapROS::visCallback, this);

  // Setup publishers for map visualization (shared �?? merged map output)
  occupied_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupied", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/unknown", 10);
  free_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/free", 10);
  occupied_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupied_inflate", 10);

  object_grid_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/occupancy_object", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/grid_map/esdf", 10);
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

  // Setup per-agent subscribers with namespaced topics
  // Depth + pose synchronizer and object cloud + ITM score per agent
  std::string sensor_pose_topic("/map_ros/pose");
  std::string depth_topic("/map_ros/depth");
  node_.param("sensor_pose_topic", sensor_pose_topic, sensor_pose_topic);
  node_.param("depth_topic", depth_topic, depth_topic);

  for (int id = 0; id < NUM_AGENTS_; ++id) {
    // Derive per-agent topic name from base template
    // If topic contains "agent_X", replace X; otherwise append /agent_X suffix
    auto makeAgentTopic = [id](const std::string& base) -> std::string {
      size_t pos = base.rfind("agent_");
      if (pos != std::string::npos) {
        // Find the end of the digit sequence after "agent_"
        size_t num_start = pos + 6; // length of "agent_"
        size_t num_end = num_start;
        while (num_end < base.size() && std::isdigit(base[num_end])) {
          num_end++;
        }
        // Replace only the agent number, preserving the suffix
        return base.substr(0, pos) + "agent_" + std::to_string(id) + base.substr(num_end);
      }
      // No agent marker found �?? append suffix
      return base + "/agent_" + std::to_string(id);
    };

    std::string agent_depth = makeAgentTopic(depth_topic);
    std::string agent_pose = makeAgentTopic(sensor_pose_topic);

    ROS_INFO("Agent %d: subscribing depth=%s pose=%s", id,
        agent_depth.c_str(), agent_pose.c_str());

    // Depth + Pose synchronizer
    // Asymmetric queues: depth is low-frequency (~10 Hz), odom is high-frequency (100-200 Hz)
    // Pose queue needs larger buffer to survive callback backlog between depth messages
    depth_sub_.push_back(
        shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
            new message_filters::Subscriber<sensor_msgs::Image>(
                node_, agent_depth, 30)));
    pose_sub_.push_back(
        shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>>(
            new message_filters::Subscriber<nav_msgs::Odometry>(
                node_, agent_pose, 60)));

    sync_image_pose_.push_back(SynchronizerImagePose(
        new message_filters::Synchronizer<SyncPolicyImagePose>(
            SyncPolicyImagePose(40), *depth_sub_.back(), *pose_sub_.back())));
    sync_image_pose_.back()->setMaxIntervalDuration(ros::Duration(0.05));
    sync_image_pose_.back()->registerCallback(
        boost::bind(&MapROS::depthPoseCallback, this, id, _1, _2));

    // Object detection and ITM score subscribers
    std::string agent_itm = "/blip2/agent_" + std::to_string(id) + "/cosine_score";
    std::string agent_cld = "/detector/agent_" + std::to_string(id) + "/clouds_with_scores";

    detected_object_cloud_sub_.push_back(
        node_.subscribe(agent_cld, 10,
            boost::function<void(const plan_env::MultipleMasksWithConfidenceConstPtr&)>(
                [this, id](const plan_env::MultipleMasksWithConfidenceConstPtr& msg) {
                    return detectedObjectCloudCallback(id, msg);
                })));
    itm_score_sub_.push_back(
        node_.subscribe(agent_itm, 10,
            boost::function<void(const std_msgs::Float64ConstPtr&)>(
                [this, id](const std_msgs::Float64ConstPtr& msg) {
                    return itmScoreCallback(id, msg);
                })));

    std::string debug_topic =
        "/semantic_observability/agent_" + std::to_string(id) + "/evidence_debug";
    semantic_evidence_debug_pub_.push_back(
        node_.advertise<plan_env::SemanticEvidenceDebug>(debug_topic, 30, true));
    DetectedObject init_detection;
    init_detection.cloud.reset(new PointCloud3D());
    init_detection.score = 0.0;
    init_detection.label = -1;
    init_detection.mask_scale = 0.0;
    init_detection.distance = 0.0;
    init_detection.view_angle = 0.0;
    init_detection.camera_height = agents_[id].camera_pos_(2);
    init_detection.mu_v = 1.0;
    init_detection.sigma_v = 0.35;
    init_detection.agent_id = id;
    publishSemanticEvidenceDebug(
        id, init_detection, -1, "init", "waiting", "waiting_for_detection_message");
  }

  // Initialize object tracking variables (shared)
  local_updated_ = false;
  esdf_need_update_ = false;
}

void MapROS::visCallback(const ros::TimerEvent& /*event*/)
{
  vis_timer_.stop();

  // All publish functions read shared map data �?? protect with mutex
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    publishOccupied();
    publishInfOccupied();
    publishObjectMap();
    publishUnknown();
    publishFree();
    publishValueMap();
    publishESDFMap();
  }

  vis_timer_.start();
}

void MapROS::itmScoreCallback(int agent_id, const std_msgs::Float64ConstPtr& msg)
{
  if (agent_id < 0 || agent_id >= NUM_AGENTS_) return;
  agents_[agent_id].itm_score_ = msg->data;
}

void MapROS::detectedObjectCloudCallback(int agent_id, const plan_env::MultipleMasksWithConfidenceConstPtr& msg)
{
  if (agent_id < 0 || agent_id >= NUM_AGENTS_) return;
  AgentState& agent = agents_[agent_id];

  // Validate message structure consistency. mask_scales is required for normal
  // Mission 2 operation, but keep a fallback for old/debug publishers.
  if (!(msg->confidence_scores.size() == msg->point_clouds.size() &&
          msg->confidence_scores.size() == msg->label_indices.size())) {
    ROS_ERROR("[Bug] The MultipleMasksWithConfidence msg is wrong!!!");
    return;
  }
  const bool has_mask_scales = msg->confidence_scores.size() == msg->mask_scales.size();
  if (!has_mask_scales) {
    ROS_WARN_THROTTLE(5.0,
        "[SemanticObservability] mask_scales size mismatch: detections=%lu mask_scales=%lu. "
        "Using fallback mask_scale=1.0 for this message.",
        msg->confidence_scores.size(), msg->mask_scales.size());
  }

  auto t1 = ros::Time::now();

  // Check camera orientation �?? only process when looking down
  Eigen::Vector3d euler =
      agent.camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order: yaw, roll, pitch
  if (euler[2] < 0)
    euler[2] += M_PI;
  double camera_pitch = euler[2];
  if (loose_semantic_evidence_debug_) {
    if (msg->confidence_scores.empty()) {
      DetectedObject empty_detection;
      empty_detection.cloud.reset(new PointCloud3D());
      empty_detection.score = 0.0;
      empty_detection.label = -1;
      empty_detection.mask_scale = 0.0;
      empty_detection.distance = 0.0;
      empty_detection.view_angle = 0.0;
      empty_detection.camera_height = agent.camera_pos_(2);
      empty_detection.mu_v = 1.0;
      empty_detection.sigma_v = 0.35;
      empty_detection.agent_id = agent_id;
      publishSemanticEvidenceDebug(
          agent_id, empty_detection, -1, "input", "empty", "detection_message_empty");
    }
    for (int i = 0; i < (int)msg->confidence_scores.size(); i++) {
      PointCloud3D::Ptr raw_cloud(new PointCloud3D());
      pcl::fromROSMsg(msg->point_clouds[i], *raw_cloud);

      double object_distance = 0.0;
      double view_angle = 0.0;
      if (!raw_cloud->points.empty()) {
        Eigen::Vector3d object_center = Eigen::Vector3d::Zero();
        for (const auto& object_pt : raw_cloud->points) {
          object_center += Eigen::Vector3d(object_pt.x, object_pt.y, object_pt.z);
        }
        object_center /= (double)raw_cloud->points.size();
        object_distance = (object_center - agent.camera_pos_).norm();

        if (object_distance > 1e-6) {
          Eigen::Vector3d target_dir = (object_center - agent.camera_pos_) / object_distance;
          Eigen::Vector3d optical_axis = agent.camera_q_.toRotationMatrix() * Eigen::Vector3d::UnitZ();
          double cos_theta = optical_axis.normalized().dot(target_dir);
          cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
          view_angle = std::acos(cos_theta);
        }
      }

      DetectedObject raw_detection;
      raw_detection.cloud = raw_cloud;
      raw_detection.score = msg->confidence_scores[i];
      raw_detection.label = msg->label_indices[i];
      raw_detection.mask_scale = has_mask_scales ? msg->mask_scales[i] : 1.0;
      raw_detection.distance = object_distance;
      raw_detection.view_angle = view_angle;
      raw_detection.camera_height = agent.camera_pos_(2);
      raw_detection.mu_v = 1.0;
      raw_detection.sigma_v = 0.35;
      raw_detection.agent_id = agent_id;
      publishSemanticEvidenceDebug(
          agent_id, raw_detection, -1, "input", "received", "raw_detection_before_filters");
    }
  }

  if (camera_pitch < 1.5 && !loose_semantic_evidence_debug_) {  // Skip if camera not tilted down enough
    DetectedObject pitch_detection;
    pitch_detection.cloud.reset(new PointCloud3D());
    pitch_detection.score = 0.0;
    pitch_detection.label = -1;
    pitch_detection.mask_scale = 0.0;
    pitch_detection.distance = 0.0;
    pitch_detection.view_angle = 0.0;
    pitch_detection.camera_height = agent.camera_pos_(2);
    pitch_detection.mu_v = 1.0;
    pitch_detection.sigma_v = 0.35;
    pitch_detection.agent_id = agent_id;
    publishSemanticEvidenceDebug(
        agent_id, pitch_detection, -1, "pitch_gate", "rejected", "camera_pitch_below_1.5");
    return;
  }

  // Backup previous per-agent over-depth object cloud for consistency tracking
  auto last_over_depth_cloud =
      boost::make_shared<PointCloud3D>(*agent.over_depth_object_cloud_);
  agent.over_depth_object_cloud_.reset(new PointCloud3D());

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
    auto mask_scale = has_mask_scales ? msg->mask_scales[i] : 1.0;
    if (label < 0) {
      ROS_WARN("[SemanticObservability] Ignore detection with invalid label %d", label);
      DetectedObject invalid_detection;
      invalid_detection.cloud.reset(new PointCloud3D());
      pcl::fromROSMsg(cloud, *invalid_detection.cloud);
      invalid_detection.score = confidence_score;
      invalid_detection.label = label;
      invalid_detection.mask_scale = mask_scale;
      invalid_detection.camera_height = agent.camera_pos_(2);
      invalid_detection.agent_id = agent_id;
      publishSemanticEvidenceDebug(
          agent_id, invalid_detection, -1, "label_filter", "rejected", "invalid_label");
      continue;
    }

    // Convert ROS message to PCL point cloud
    PointCloud3D::Ptr single_object_cloud(new PointCloud3D());
    pcl::fromROSMsg(cloud, *single_object_cloud);
    *all_object_cloud += *single_object_cloud;

    // Apply voxel grid downsampling
    voxel_filter.setInputCloud(single_object_cloud);
    voxel_filter.setLeafSize(0.04f, 0.04f, 0.06f);
    voxel_filter.filter(*single_object_cloud);

    // Filter out points beyond sensor accuracy range
    PointCloud3D::Ptr tmp_object_cloud(new PointCloud3D());
    PointCloud3D::Ptr over_depth_object_cloud(new PointCloud3D());
    for (auto object_pt : single_object_cloud->points) {
      Eigen::Vector3d object_pt3d = Eigen::Vector3d(object_pt.x, object_pt.y, object_pt.z);
      if ((object_pt3d - agent.camera_pos_).norm() > depth_filter_maxdist_ - 0.10) {
        // Store over-depth points for target objects (label == 0)
        if (label == 0)
          over_depth_object_cloud->points.push_back(object_pt);
        continue;
      }
      tmp_object_cloud->points.push_back(object_pt);
    }
    single_object_cloud = tmp_object_cloud;

    // Skip objects entirely beyond valid range
    if (single_object_cloud->points.empty()) {
      if (!over_depth_object_cloud->points.empty()) {
        ROS_ERROR("Have all over depth object cloud!!!!");
        *agent.over_depth_object_cloud_ += *over_depth_object_cloud;
      }
      DetectedObject rejected_object;
      rejected_object.cloud = over_depth_object_cloud;
      rejected_object.score = confidence_score;
      rejected_object.label = label;
      rejected_object.mask_scale = mask_scale;
      rejected_object.camera_height = agent.camera_pos_(2);
      rejected_object.agent_id = agent_id;
      publishSemanticEvidenceDebug(
          agent_id, rejected_object, -1, "depth_filter", "rejected", "all_points_over_depth");
      continue;
    }

    // DBSCAN clustering
    single_object_cloud = dbscan(single_object_cloud, 0.12f, 10);
    if (single_object_cloud == nullptr) {
      ROS_ERROR("After DBSCAN, no point cloud cluster!!");
      DetectedObject rejected_object;
      rejected_object.cloud.reset(new PointCloud3D());
      rejected_object.score = confidence_score;
      rejected_object.label = label;
      rejected_object.mask_scale = mask_scale;
      rejected_object.camera_height = agent.camera_pos_(2);
      rejected_object.agent_id = agent_id;
      publishSemanticEvidenceDebug(
          agent_id, rejected_object, -1, "dbscan", "rejected", "no_cluster_found");
      continue;
    }

    if (single_object_cloud->points.empty()) {
      ROS_ERROR("Single object point cloud is empty!!!");
      DetectedObject rejected_object;
      rejected_object.cloud = single_object_cloud;
      rejected_object.score = confidence_score;
      rejected_object.label = label;
      rejected_object.mask_scale = mask_scale;
      rejected_object.camera_height = agent.camera_pos_(2);
      rejected_object.agent_id = agent_id;
      publishSemanticEvidenceDebug(
          agent_id, rejected_object, -1, "dbscan", "rejected", "cluster_empty");
      continue;
    }

    Eigen::Vector3d object_center = Eigen::Vector3d::Zero();
    for (const auto& object_pt : single_object_cloud->points) {
      object_center += Eigen::Vector3d(object_pt.x, object_pt.y, object_pt.z);
    }
    object_center /= (double)single_object_cloud->points.size();
    double object_distance = (object_center - agent.camera_pos_).norm();

    double view_angle = 0.0;
    if (object_distance > 1e-6) {
      Eigen::Vector3d target_dir = (object_center - agent.camera_pos_) / object_distance;
      Eigen::Vector3d optical_axis = agent.camera_q_.toRotationMatrix() * Eigen::Vector3d::UnitZ();
      double cos_theta = optical_axis.normalized().dot(target_dir);
      cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
      view_angle = std::acos(cos_theta);
    }

    std::string prior_base = "/semantic_prior/label_" + std::to_string(label);
    double mu_v = 1.0;
    double sigma_v = 0.35;
    ros::param::param(prior_base + "/mu_v", mu_v, mu_v);
    ros::param::param(prior_base + "/sigma_v", sigma_v, sigma_v);

    *filtered_all_object_cloud += *single_object_cloud;
    DetectedObject detected_object;
    detected_object.cloud = single_object_cloud;
    detected_object.score = confidence_score;
    detected_object.label = label;
    detected_object.mask_scale = mask_scale;
    detected_object.distance = object_distance;
    detected_object.view_angle = view_angle;
    detected_object.camera_height = agent.camera_pos_(2);
    detected_object.mu_v = mu_v;
    detected_object.sigma_v = sigma_v;
    detected_object.agent_id = agent_id;
    detected_objects.push_back(detected_object);
  }

  {
    std::lock_guard<std::mutex> lock(map_mutex_);

    // Maintain per-agent over-depth consistency, then merge into shared cloud
    if (agent.continue_over_depth_count_ == -1 &&
        !agent.over_depth_object_cloud_->points.empty())
      agent.continue_over_depth_count_ = 0;
    else if (agent.continue_over_depth_count_ <= 4 && agent.continue_over_depth_count_ >= 0) {
      agent.continue_over_depth_count_++;
      agent.over_depth_object_cloud_ = last_over_depth_cloud;
    }
    else {
      agent.continue_over_depth_count_ = -1;
    }

    // Merge all agents' over-depth clouds into the shared visualization cloud
    map_->object_map2d_->over_depth_object_cloud_.reset(new PointCloud3D());
    for (int i = 0; i < NUM_AGENTS_; ++i)
      *map_->object_map2d_->over_depth_object_cloud_ += *agents_[i].over_depth_object_cloud_;

    // Publish visualization
    publishPointCloud(filtered_object_cloud_pub_, filtered_all_object_cloud);
    publishPointCloud(all_object_cloud_pub_, all_object_cloud);
    publishPointCloud(over_depth_object_cloud_pub_, map_->object_map2d_->over_depth_object_cloud_);

    // Update object map
    *map_->object_map2d_->all_object_clouds_ = *filtered_all_object_cloud;
    vector<int> detected_object_cluster_ids;
    map_->inputObjectCloud2D(detected_objects, detected_object_cluster_ids);
    for (int i = 0; i < (int)detected_objects.size() &&
                    i < (int)detected_object_cluster_ids.size(); ++i) {
      const int cluster_id = detected_object_cluster_ids[i];
      publishSemanticEvidenceDebug(agent_id, detected_objects[i], cluster_id,
          "map_update", cluster_id >= 0 ? "accepted" : "rejected",
          cluster_id >= 0 ? "" : "no_occupied_object_cells");
    }

    // Extract observation objects not detected by vision
    getObservationObjectsCloud(agent_id, detected_object_cluster_ids);
  }

  double object_map_process_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(
      10.0, "[Calculating Time] Object Map process time = %.3f s", object_map_process_time);
}

void MapROS::publishSemanticEvidenceDebug(
    int agent_id, const DetectedObject& detected_object, int object_cluster_id,
    const std::string& stage, const std::string& status, const std::string& reason)
{
  if (agent_id < 0 || agent_id >= (int)semantic_evidence_debug_pub_.size())
    return;
  SemanticEvidenceSnapshot snapshot;
  if (map_ != nullptr && map_->object_map2d_ != nullptr)
    map_->object_map2d_->getSemanticEvidenceConfig(snapshot);
  bool has_snapshot = false;
  if (object_cluster_id >= 0 && map_ != nullptr && map_->object_map2d_ != nullptr) {
    has_snapshot = map_->object_map2d_->getSemanticEvidenceSnapshot(
        object_cluster_id, detected_object.label, snapshot);
  }

  const int raw_point_count = detected_object.cloud ? detected_object.cloud->points.size() : 0;
  double observation_rho = semantic_observability::observability(detected_object.camera_height,
      detected_object.mu_v, detected_object.sigma_v, detected_object.distance,
      detected_object.view_angle, detected_object.mask_scale, snapshot.lambda_d, snapshot.r0,
      snapshot.mask_sigmoid_k);
  double observation_evidence = semantic_observability::saturatedEvidence(
      observation_rho, detected_object.score, 1, snapshot.beta);

  plan_env::SemanticEvidenceDebug msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_id_;
  msg.agent_id = agent_id;
  msg.cluster_id = has_snapshot ? snapshot.cluster_id : object_cluster_id;
  msg.label = has_snapshot ? snapshot.label : detected_object.label;
  msg.best_label = has_snapshot ? snapshot.best_label : -1;
  msg.stage = stage;
  msg.status = status;
  msg.reason = reason;
  msg.accepted = status == "accepted";
  msg.use_semantic_observability = snapshot.use_semantic_observability;
  msg.lambda_d = snapshot.lambda_d;
  msg.r0 = snapshot.r0;
  msg.mask_sigmoid_k = snapshot.mask_sigmoid_k;
  msg.beta = snapshot.beta;
  msg.min_semantic_evidence = snapshot.min_semantic_evidence;
  msg.min_observation_num = snapshot.min_observation_num;
  msg.camera_height = detected_object.camera_height;
  msg.mu_v = detected_object.mu_v;
  msg.sigma_v = detected_object.sigma_v;
  msg.distance = detected_object.distance;
  msg.view_angle = detected_object.view_angle;
  msg.mask_scale = detected_object.mask_scale;
  msg.raw_confidence = detected_object.score;
  msg.fused_confidence = has_snapshot ? snapshot.fused_confidence : 0.0;
  msg.observation_num = has_snapshot ? snapshot.observation_num : 0;
  msg.observation_cloud_sum =
      has_snapshot ? snapshot.observation_cloud_sum : raw_point_count;
  msg.observability = has_snapshot ? snapshot.observability : observation_rho;
  msg.quality_evidence = has_snapshot ? snapshot.quality_evidence : observation_evidence;
  msg.target_quality_evidence = has_snapshot ? snapshot.target_quality_evidence : 0.0;
  msg.target_fused_confidence = has_snapshot ? snapshot.target_fused_confidence : 0.0;
  msg.target_observation_num = has_snapshot ? snapshot.target_observation_num : 0;
  msg.target_passes_threshold = has_snapshot ? snapshot.target_passes_threshold : false;
  msg.target_is_best_label = has_snapshot ? snapshot.target_is_best_label : false;
  msg.top_label = has_snapshot ? snapshot.top_label : detected_object.label;
  msg.top_quality_evidence =
      has_snapshot ? snapshot.top_quality_evidence : observation_evidence;
  msg.top_fused_confidence = has_snapshot ? snapshot.top_fused_confidence : detected_object.score;
  msg.top_observation_num = has_snapshot ? snapshot.top_observation_num : 1;
  msg.top_observation_cloud_sum =
      has_snapshot ? snapshot.top_observation_cloud_sum : raw_point_count;
  msg.top_observability = has_snapshot ? snapshot.top_observability : observation_rho;
  msg.top_distance = has_snapshot ? snapshot.top_distance : detected_object.distance;
  msg.top_view_angle = has_snapshot ? snapshot.top_view_angle : detected_object.view_angle;
  msg.top_mask_scale = has_snapshot ? snapshot.top_mask_scale : detected_object.mask_scale;
  msg.top_passes_threshold = has_snapshot ? snapshot.top_passes_threshold : false;
  msg.second_label = has_snapshot ? snapshot.second_label : -1;
  msg.second_quality_evidence = has_snapshot ? snapshot.second_quality_evidence : 0.0;
  msg.second_fused_confidence = has_snapshot ? snapshot.second_fused_confidence : 0.0;
  msg.second_observation_num = has_snapshot ? snapshot.second_observation_num : 0;
  msg.second_observation_cloud_sum =
      has_snapshot ? snapshot.second_observation_cloud_sum : 0;
  msg.second_observability = has_snapshot ? snapshot.second_observability : 0.0;
  msg.second_distance = has_snapshot ? snapshot.second_distance : 0.0;
  msg.second_view_angle = has_snapshot ? snapshot.second_view_angle : 0.0;
  msg.second_mask_scale = has_snapshot ? snapshot.second_mask_scale : 0.0;
  msg.second_passes_threshold = has_snapshot ? snapshot.second_passes_threshold : false;
  msg.top_second_abs_diff = has_snapshot ? snapshot.top_second_abs_diff
                                         : std::abs(observation_evidence);
  msg.target_is_top_label = has_snapshot ? snapshot.target_is_top_label
                                         : detected_object.label == 0;

  semantic_evidence_debug_pub_[agent_id].publish(msg);
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/)
{
  if (!esdf_need_update_)
    return;

  esdf_timer_.stop();

  auto t1 = ros::Time::now();

  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    map_->updateESDFMap();
  }

  esdf_need_update_ = false;
  double esdf_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] ESDF Map process time = %.3f s", esdf_time);

  esdf_timer_.start();
}

void MapROS::depthPoseCallback(
    int agent_id, const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose)
{
  if (agent_id < 0 || agent_id >= NUM_AGENTS_) return;
  AgentState& agent = agents_[agent_id];

  // Extract camera pose from odometry message
  agent.camera_pos_(0) = pose->pose.pose.position.x;
  agent.camera_pos_(1) = pose->pose.pose.position.y;
  agent.camera_pos_(2) = pose->pose.pose.position.z;
  agent.camera_q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
      pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);

  // Calculate camera yaw for value map updates
  Eigen::Vector3d euler =
      agent.camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);
  double camera_yaw = euler[0];
  Eigen::Vector2d camera_pos = Eigen::Vector2d(agent.camera_pos_(0), agent.camera_pos_(1));

  // Skip if camera outside map bounds
  if (!map_->isInMap(camera_pos))
    return;

  // Convert depth image format
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 255.0);
  cv_ptr->image.copyTo(*agent.depth_image_);

  auto t1 = ros::Time::now();

  processDepthImage(agent_id);
  filterPointCloudToXY(agent_id);

  // Update shared map data (requires mutex �?? write to shared map)
  std::lock_guard<std::mutex> lock(map_mutex_);

  // Virtual ground ground points collected in filterPointCloudToXY
  if (!agent.under_ground_cloud2d_->empty())
    map_->inputVirtualGround(agent.under_ground_cloud2d_);

  vector<Eigen::Vector2i> free_grids;
  dilateGrids(free_grids, 1);
  map_->inputDepthCloud2D(agent.filtered_depth_cloud2d_, agent.camera_pos_, free_grids);
  double process_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] Grid Map process time = %.3f s", process_time);

  // Update semantic value map if ITM score is available
  if (agent.itm_score_ != -1.0)
    map_->value_map_->updateValueMap(camera_pos, camera_yaw, free_grids, agent.itm_score_);
  double value_map_time = (ros::Time::now() - t1).toSec();
  ROS_INFO_THROTTLE(50.0, "[Calculating Time] Value Map process time = %.3f s", value_map_time);

  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }
}

void MapROS::processDepthImage(int agent_id)
{
  AgentState& agent = agents_[agent_id];
  agent.proj_points_cnt_ = 0;

  uint16_t* row_ptr;
  int cols = agent.depth_image_->cols;
  int rows = agent.depth_image_->rows;
  double depth;
  Eigen::Matrix3d camera_r = agent.camera_q_.toRotationMatrix();

  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    row_ptr = agent.depth_image_->ptr<uint16_t>(v) + depth_filter_margin_;
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {
      depth = (*row_ptr) * (1.0 / k_depth_scaling_factor_) *
                  (depth_filter_maxdist_ - depth_filter_mindist_) +
              depth_filter_mindist_;
      row_ptr = row_ptr + skip_pixel_;

      if (depth > depth_filter_maxdist_)
        depth = depth_filter_maxdist_;
      else if (depth < depth_filter_mindist_)
        continue;

      Eigen::Vector3d pt_cur(
          (u - cx_) * depth / fx_,
          (v - cy_) * depth / fy_,
          depth);

      agent.camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);
      Eigen::Vector3d pt_world = camera_r * pt_cur + agent.camera_pos_;
      agent.depth_cloud_->points[agent.proj_points_cnt_++] =
          Point3D(pt_world[0], pt_world[1], pt_world[2]);
    }
  }
  publishPointCloud(depth_cloud_pub_, agent.depth_cloud_);
}

void MapROS::filterPointCloudToXY(int agent_id)
{
  AgentState& agent = agents_[agent_id];
  PointCloud3D::Ptr filtered_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr down_depth_cloud_3d(new PointCloud3D());
  PointCloud3D::Ptr under_ground_cloud_3d(new PointCloud3D());
  PointCloud2D::Ptr under_ground_cloud_2d(new PointCloud2D());

  agent.filtered_depth_cloud2d_->clear();

  // Downsample
  pcl::VoxelGrid<Point3D> voxel_filter;
  voxel_filter.setInputCloud(agent.depth_cloud_);
  voxel_filter.setLeafSize(0.04f, 0.04f, 0.1f);
  voxel_filter.filter(*down_depth_cloud_3d);

  double cur_floor_height = 0.0;
  double virtual_ground = virtual_ground_height_;

  for (int i = 0; i < (int)down_depth_cloud_3d->points.size(); i++) {
    Point3D pt;
    pt.x = down_depth_cloud_3d->points[i].x;
    pt.y = down_depth_cloud_3d->points[i].y;
    pt.z = down_depth_cloud_3d->points[i].z;

    if (down_depth_cloud_3d->points[i].z < cur_floor_height + virtual_ground)
      under_ground_cloud_3d->points.push_back(pt);
    else if (down_depth_cloud_3d->points[i].z > cur_floor_height + filter_min_height_ &&
             down_depth_cloud_3d->points[i].z < cur_floor_height + filter_max_height_)
      filtered_cloud_3d->points.push_back(pt);
  }

  pcl::RadiusOutlierRemoval<Point3D> outrem;

  if (!filtered_cloud_3d->points.empty()) {
    outrem.setInputCloud(filtered_cloud_3d);
    outrem.setRadiusSearch(0.3);
    outrem.setMinNeighborsInRadius(35);
    outrem.filter(*filtered_cloud_3d);
  }

  publishPointCloud(filtered_depth_cloud_pub_, filtered_cloud_3d);

  // Project 3D to 2D
  for (auto pt : filtered_cloud_3d->points) {
    Point2D pt_xy;
    pt_xy.x = pt.x;
    pt_xy.y = pt.y;
    agent.filtered_depth_cloud2d_->points.push_back(pt_xy);
  }

  if (!under_ground_cloud_3d->points.empty()) {
    outrem.setInputCloud(under_ground_cloud_3d);
    outrem.setRadiusSearch(0.21);
    outrem.setMinNeighborsInRadius(40);
    outrem.filter(*under_ground_cloud_3d);
  }

  Eigen::Vector3d euler =
      agent.camera_q_.toRotationMatrix().eulerAngles(2, 1, 0);
  if (euler[2] < 0)
    euler[2] += M_PI;
  double camera_pitch = euler[2];

  if (camera_pitch > 1.5 && !under_ground_cloud_3d->points.empty()) {
    agent.under_ground_cloud2d_->clear();
    for (auto pt : under_ground_cloud_3d->points) {
      Eigen::Vector3d pt_pos = Eigen::Vector3d(pt.x, pt.y, pt.z);
      Eigen::Vector2d ground_pos;

      if (interpolateLineAtZ(pt_pos, agent.camera_pos_, cur_floor_height + virtual_ground, ground_pos)) {
        Point2D pt_xy;
        pt_xy.x = ground_pos(0);
        pt_xy.y = ground_pos(1);
        agent.filtered_depth_cloud2d_->points.push_back(pt_xy);
        agent.under_ground_cloud2d_->points.push_back(pt_xy);
      }
    }
  }
}

bool MapROS::interpolateLineAtZ(
    const Eigen::Vector3d& A, const Eigen::Vector3d& B, double target_z, Eigen::Vector2d& P)
{
  if ((A.z() - target_z) * (B.z() - target_z) > 0)
    return false;

  double t = (target_z - A.z()) / (B.z() - A.z());
  double x = A.x() + t * (B.x() - A.x());
  double y = A.y() + t * (B.y() - A.y());
  P = Eigen::Vector2d(x, y);
  return true;
}

void MapROS::getObservationObjectsCloud(int agent_id, const vector<int>& filter_object_ids)
{
  // Caller already holds map_mutex_ �?? called from within detectedObjectCloudCallback's lock scope
  AgentState& agent = agents_[agent_id];

  // Downsample depth cloud
  PointCloud3D::Ptr filtered_depth_cloud(new PointCloud3D());
  pcl::VoxelGrid<Point3D> voxel_filter;
  voxel_filter.setInputCloud(agent.depth_cloud_);
  voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
  voxel_filter.filter(*filtered_depth_cloud);

  // Get bounding boxes
  vector<Vector3d> bmins, bmaxs;
  map_->object_map2d_->getObjectBoxes(bmins, bmaxs);
  vector<char> filter_object_flag(bmins.size(), 0);
  for (auto filter_object_id : filter_object_ids) {
    if (filter_object_id >= 0 && filter_object_id < (int)filter_object_flag.size())
      filter_object_flag[filter_object_id] = 1;
  }

  pcl::CropBox<Point3D> crop_box_filter;
  crop_box_filter.setInputCloud(filtered_depth_cloud);
  vector<pcl::shared_ptr<PointCloud3D>> observation_clouds;

  for (int i = 0; i < (int)bmins.size(); i++) {
    PointCloud3D::Ptr cloud_filtered(new PointCloud3D);
    if (filter_object_flag[i])
      observation_clouds.push_back(cloud_filtered);
    else {
      double inf = 0.2f;
      Eigen::Vector4f min_point(bmins[i][0] - inf, bmins[i][1] - inf, bmins[i][2] - inf, 1.0);
      Eigen::Vector4f max_point(bmaxs[i][0] + inf, bmaxs[i][1] + inf, bmaxs[i][2] + inf, 1.0);
      crop_box_filter.setMin(min_point);
      crop_box_filter.setMax(max_point);
      crop_box_filter.filter(*cloud_filtered);
      observation_clouds.push_back(cloud_filtered);
    }
  }

  map_->object_map2d_->inputObservationObjectsCloud(observation_clouds, max(0.0, agent.itm_score_));
}

PointCloud3D::Ptr MapROS::dbscan(const PointCloud3D::Ptr& cloud, double eps, int minPts)
{
  if (cloud->empty()) {
    ROS_ERROR("[DBSCAN] Input cloud is empty!");
    return nullptr;
  }

  pcl::search::KdTree<Point3D>::Ptr tree(new pcl::search::KdTree<Point3D>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;

  pcl::EuclideanClusterExtraction<Point3D> ec;
  ec.setClusterTolerance(eps);
  ec.setMinClusterSize(minPts);
  ec.setMaxClusterSize(cloud->points.size());
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  if (cluster_indices.empty()) {
    ROS_WARN("[DBSCAN] No clusters found!");
    return nullptr;
  }

  int largest_cluster_index = -1;
  size_t max_size = 0;
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    if (cluster_indices[i].indices.size() > max_size) {
      max_size = cluster_indices[i].indices.size();
      largest_cluster_index = i;
    }
  }

  PointCloud3D::Ptr largest_cluster(new PointCloud3D);
  for (int idx : cluster_indices[largest_cluster_index].indices)
    largest_cluster->points.push_back(cloud->points[idx]);
  return largest_cluster;
}

void MapROS::dilateGrids(std::vector<Eigen::Vector2i>& grids, int dilation_radius)
{
  if (grids.empty() || dilation_radius <= 0)
    return;

  std::unordered_set<uint64_t> dilated_grid_set;

  auto hash_grid = [](int x, int y) -> uint64_t {
    return (static_cast<uint64_t>(x) << 32) | static_cast<uint32_t>(y);
  };

  std::vector<Eigen::Vector2i> dilation_template;
  for (int dx = -dilation_radius; dx <= dilation_radius; ++dx) {
    for (int dy = -dilation_radius; dy <= dilation_radius; ++dy) {
      if (dx * dx + dy * dy <= dilation_radius * dilation_radius) {
        dilation_template.emplace_back(dx, dy);
      }
    }
  }

  for (const auto& grid : grids) {
    for (const auto& offset : dilation_template) {
      Eigen::Vector2i new_grid = grid + offset;

      Eigen::Vector2d new_pos;
      map_->indexToPos(new_grid, new_pos);
      if (map_->isInMap(new_pos)) {
        dilated_grid_set.insert(hash_grid(new_grid.x(), new_grid.y()));
      }
    }
  }

  grids.clear();
  grids.reserve(dilated_grid_set.size());

  for (const auto& grid_hash : dilated_grid_set) {
    int x = static_cast<int>(grid_hash >> 32);
    int y = static_cast<int>(grid_hash & 0xFFFFFFFF);
    grids.emplace_back(x, y);
  }
}

// ── Visualization publishing functions (all read shared map_ state) ─────────────
// Caller must hold map_mutex_. These are invoked from visCallback under lock.
// ───────────────────────────────────────────────────────────────────────────────

void MapROS::publishObjectMap()
{
  PointCloud3D cloud;
  for (int x = map_->md_->update_min_(0); x < map_->md_->update_max_(0); ++x)
    for (int y = map_->md_->update_min_(1); y < map_->md_->update_max_(1); ++y) {
      if (map_->object_map2d_->getObjectGrid(map_->toAddress(x, y)) == 1) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        Point3D pt;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.05;
        cloud.push_back(pt);
      }
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  object_grid_pub_.publish(cloud_msg);
}

void MapROS::publishOccupied()
{
  PointCloud3D cloud;
  for (int x = map_->md_->update_min_(0); x < map_->md_->update_max_(0); ++x)
    for (int y = map_->md_->update_min_(1); y < map_->md_->update_max_(1); ++y) {
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        Point3D pt;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.0;
        cloud.push_back(pt);
      }
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  occupied_pub_.publish(cloud_msg);
}

void MapROS::publishInfOccupied()
{
  PointCloud3D cloud;
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
        continue;
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        Point3D pt;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.0;
        cloud.push_back(pt);
      }
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  occupied_inflate_pub_.publish(cloud_msg);
}

void MapROS::publishUnknown()
{
  PointCloud3D cloud;
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
        continue;
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3) {
        Eigen::Vector2d pos;
        map_->indexToPos(Eigen::Vector2i(x, y), pos);
        Point3D pt;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = 0.0;
        cloud.push_back(pt);
      }
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

void MapROS::publishFree()
{
  PointCloud3D cloud;
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
        continue;
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3)
        continue;
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
        continue;
      Eigen::Vector2d pos;
      map_->indexToPos(Eigen::Vector2i(x, y), pos);
      Point3D pt;
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = 0.0;
      cloud.push_back(pt);
    }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  free_pub_.publish(cloud_msg);
}

void MapROS::publishValueMap()
{
  Eigen::Vector2i min_cut = map_->md_->update_min_;
  Eigen::Vector2i max_cut = map_->md_->update_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  pcl::PointCloud<pcl::PointXYZI> cloud_with_intensity;
  const double min_value = 0.0;
  const double max_value = 1.0;

  for (int x = min_cut(0); x <= max_cut(0); ++x) {
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      double value = map_->value_map_->getValue(Eigen::Vector2i(x, y));
      if (value > 1e-3) {
        if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
          continue;
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3)
          continue;
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
          continue;

        Eigen::Vector2d pos_2d;
        map_->indexToPos(Eigen::Vector2i(x, y), pos_2d);
        value = std::min(value, max_value);
        value = std::max(value, min_value);
        pcl::PointXYZI pt;
        pt.x = pos_2d(0);
        pt.y = pos_2d(1);
        pt.z = 0.08;
        pt.intensity = (value - min_value) / (max_value - min_value);
        cloud_with_intensity.push_back(pt);
      }
    }
  }

  cloud_with_intensity.width = cloud_with_intensity.points.size();
  cloud_with_intensity.height = 1;
  cloud_with_intensity.is_dense = true;
  cloud_with_intensity.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_with_intensity, cloud_msg);
  value_map_pub_.publish(cloud_msg);
}

void MapROS::publishESDFMap()
{
  Eigen::Vector2i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector2i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  pcl::PointCloud<pcl::PointXYZI> cloud_with_intensity;
  const double min_dist = 0.0;
  const double max_dist = 3.0;

  for (int x = min_cut(0); x <= max_cut(0); ++x) {
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y)] == 1)
        continue;
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] < map_->mp_->clamp_min_log_ - 1e-3)
        continue;
      if (map_->md_->occupancy_buffer_[map_->toAddress(x, y)] > map_->mp_->min_occupancy_log_)
        continue;

      Eigen::Vector2d pos_2d;
      map_->indexToPos(Eigen::Vector2i(x, y), pos_2d);
      double dist = map_->getDistance(pos_2d);

      dist = std::min(dist, max_dist);
      dist = std::max(dist, min_dist);

      pcl::PointXYZI pt;
      pt.x = pos_2d(0);
      pt.y = pos_2d(1);
      pt.z = 0.08;
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud_with_intensity.push_back(pt);
    }
  }

  cloud_with_intensity.width = cloud_with_intensity.points.size();
  cloud_with_intensity.height = 1;
  cloud_with_intensity.is_dense = true;
  cloud_with_intensity.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_with_intensity, cloud_msg);
  esdf_pub_.publish(cloud_msg);
}

void MapROS::publishPointCloud(const ros::Publisher& pub, const PointCloud3D::Ptr& point_cloud)
{
  PointCloud3D cloud;
  for (int i = 0; i < (int)point_cloud->points.size(); ++i) cloud.push_back(point_cloud->points[i]);

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  pub.publish(cloud_msg);
}

}  // namespace apexnav_planner
