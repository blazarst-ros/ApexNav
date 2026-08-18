#ifndef _MAP_ROS_H
#define _MAP_ROS_H

// ROS core and message handling
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

// Custom messages and mapping components
#include <plan_env/MultipleMasksWithConfidence.h>
#include <plan_env/ObjectClusterStatusArray.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/value_map2d.h>

// OpenCV for image processing
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// Standard ROS messages
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/MarkerArray.h>

// PCL for point cloud processing
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/common.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/conditional_removal.h>
#include <unordered_set>
#include <mutex>

// Type aliases for convenience
using std::shared_ptr;
using Point3D = pcl::PointXYZ;                  ///< 3D point type for PCL operations
using PointCloud3D = pcl::PointCloud<Point3D>;  ///< 3D point cloud type
using Point2D = pcl::PointXY;                   ///< 2D point type for mapping
using PointCloud2D = pcl::PointCloud<Point2D>;  ///< 2D point cloud type for occupancy grid

namespace apexnav_planner {
class SDFMap2D;

static constexpr int OVER_DEPTH_CACHE_MAX_MISSING_FRAMES = 8;

/// Per-agent sensor state. Each agent has independent camera pose, depth buffer,
/// and ITM score, but all agents write to the shared SDFMap2D + ObjectMap2D.
struct AgentState {
  Eigen::Vector3d camera_pos_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond camera_q_ = Eigen::Quaterniond::Identity();
  unique_ptr<cv::Mat> depth_image_;
  int proj_points_cnt_ = 0;
  PointCloud3D::Ptr depth_cloud_;             ///< Raw 3D point cloud from depth sensor
  PointCloud2D::Ptr filtered_depth_cloud2d_;  ///< Filtered 2D cloud for occupancy mapping
  PointCloud2D::Ptr under_ground_cloud2d_;    ///< Virtual ground ground points for deferred map write
  PointCloud3D::Ptr over_depth_object_cloud_;  ///< Per-agent over-depth object cloud for consistency tracking
  PointCloud3D::Ptr cached_over_depth_cloud_;  ///< Last valid over-depth cloud for short loss bridging

  int over_depth_missing_frames_ = 0;  ///< Consecutive frames without fresh over-depth cloud
  double itm_score_ = -1.0;            ///< Current image-text matching score
};

class MapROS {
public:
  MapROS() = default;
  ~MapROS() = default;
  // Core interface functions
  void setMap(SDFMap2D* map);
  void init();
  void resetEpisodeState();

private:
  // ROS callback functions (each receives agent_id as first argument)
  void depthPoseCallback(                        ///< Process synchronized depth image and pose data
      int agent_id, const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose);
  void updateESDFCallback(const ros::TimerEvent& /*event*/);
  void detectedObjectCloudCallback(int agent_id, const plan_env::MultipleMasksWithConfidenceConstPtr& msg);
  void itmScoreCallback(int agent_id, const std_msgs::Float64ConstPtr& msg);
  void visCallback(const ros::TimerEvent& /*event*/);

  // Visualization publishing functions
  void publishOccupied();
  void publishInfOccupied();
  void publishFree();
  void publishUnknown();

  void publishObjectMap();
  void publishObjectVisualizations();
  void publishESDFMap();
  void publishValueMap();
  void publishPointCloud(const ros::Publisher& pub, const PointCloud3D::Ptr& point_cloud);

  // Data processing functions (all take agent_id to index agents_[agent_id])
  void processDepthImage(int agent_id);           ///< Process raw depth image into 3D point cloud
  void filterPointCloudToXY(int agent_id);        ///< Filter 3D points to 2D occupancy grid

  // Utility functions
  bool interpolateLineAtZ(
      const Eigen::Vector3d& A, const Eigen::Vector3d& B, double target_z, Eigen::Vector2d& P);
  PointCloud3D::Ptr dbscan(const PointCloud3D::Ptr& cloud, double eps, int minPts);
  void dilateGrids(std::vector<Eigen::Vector2i>& grids, int dilation_radius);

  // Core mapping interface (SHARED across all agents)
  SDFMap2D* map_;

  // The build has room for two agents; runtime config selects one or two.
  static constexpr int MAX_AGENTS_ = 2;
  int num_agents_ = 2;

  // Per-agent state
  std::vector<AgentState> agents_;

  // Mutex protecting writes to the shared map_ (SDFMap2D, ObjectMap2D, ValueMap2D)
  std::mutex map_mutex_;

  // Message synchronization types
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImagePose;  ///< Policy for synchronizing depth images with pose data
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;

  // ROS communication interfaces (shared publishers)
  ros::NodeHandle node_;

  // Per-agent subscribers and synchronizers
  vector<shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> depth_sub_;
  vector<shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>>> pose_sub_;
  vector<SynchronizerImagePose> sync_image_pose_;
  vector<ros::Subscriber> detected_object_cloud_sub_;
  vector<ros::Subscriber> itm_score_sub_;
  ros::Publisher camera_pitch_pub_[MAX_AGENTS_];  ///< Per-agent object-filter gate angle

  // ROS publishers for shared merged-map visualization
  ros::Publisher occupied_pub_, occupied_inflate_pub_, unknown_pub_, free_pub_, esdf_pub_,
      object_grid_pub_, depth_cloud_pub_, filtered_depth_cloud_pub_,
      filtered_object_cloud_pub_, all_object_cloud_pub_, over_depth_object_cloud_pub_,
      value_map_pub_, semantic_object_pub_, cluster_marker_pub_, cluster_status_pub_,
      cluster_status_image_pub_;

  // ROS timers for periodic updates (shared — operate on merged map)
  ros::Timer esdf_timer_, vis_timer_;

  // Camera intrinsic parameters (same for all agents)
  double cx_, cy_, fx_, fy_;

  // Depth filtering parameters (same for all agents)
  double depth_filter_maxdist_, depth_filter_mindist_;  ///< Valid depth range for filtering
  double filter_min_height_, filter_max_height_;        ///< Height range for obstacle detection
  int depth_filter_margin_;        ///< Margin pixels to ignore near image borders
  double k_depth_scaling_factor_;  ///< Depth value scaling factor for different sensors
  int skip_pixel_;                 ///< Pixel skip factor for processing efficiency
  std::string frame_id_;           ///< Reference frame ID for published data
  double virtual_ground_height_;   ///< Virtual ground plane offset for navigation

  // Map state flags (shared)
  bool local_updated_, esdf_need_update_;
  bool cluster_markers_need_reset_ = true;

  friend SDFMap2D;
};

}  // namespace apexnav_planner

#endif
