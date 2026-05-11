#ifndef _OBJECT_MAP2D_H_
#define _OBJECT_MAP2D_H_

// ROS and system includes
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <algorithm>
#include <cmath>

// Internal mapping components
#include <plan_env/sdf_map2d.h>
#include <plan_env/raycast2d.h>

// PCL for point cloud processing
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <unordered_map>

using Eigen::Vector2d;
using Eigen::Vector2i;
using Eigen::Vector3d;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class RayCaster2D;

namespace apexnav_planner {
class SDFMap2D;

static const char* T_COLORS[] = {
  "\033[0m",     ///< Default color [0]
  "\033[1;31m",  ///< Red color [1]
  "\033[1;32m",  ///< Green color [2]
  "\033[1;33m",  ///< Yellow color [3]
  "\033[1;34m",  ///< Blue color [4]
  "\033[1;35m",  ///< Purple color [5]
  "\033[1;36m"   ///< Cyan color [6]
};

struct DetectedObject {
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud;  ///< 3D point cloud of detected object
  double score;                                           ///< Confidence score from detector (0-1)
  int label;                                              ///< Semantic class label from detection
  double mask_scale = 1.0;                                ///< Source image mask area ratio
  double distance = 0.0;                                  ///< Camera-to-object distance
  double view_angle = 0.0;                                ///< Viewing angle term input
  double observability = 1.0;                             ///< View-conditioned observability
  double quality_evidence = 0.0;                          ///< Quality-aware semantic evidence
  double camera_height = 1.0;                             ///< Camera height for this observation
  double mu_v = 1.0;                                      ///< LLM prior recommended height
  double sigma_v = 0.35;                                  ///< LLM prior height tolerance
  int agent_id = -1;                                      ///< Source observer agent id
};

struct SemanticEvidenceSnapshot {
  int cluster_id = -1;
  int label = -1;
  int best_label = -1;
  bool use_semantic_observability = true;
  double lambda_d = 0.0;
  double r0 = 0.007;
  double mask_sigmoid_k = 300.0;
  double beta = 0.0;
  double min_semantic_evidence = 0.0;
  int min_observation_num = 0;
  double fused_confidence = 0.0;
  int observation_num = 0;
  int observation_cloud_sum = 0;
  double observability = 0.0;
  double quality_evidence = 0.0;
  double target_quality_evidence = 0.0;
  double target_fused_confidence = 0.0;
  int target_observation_num = 0;
  bool target_passes_threshold = false;
  bool target_is_best_label = false;
  int top_label = -1;
  double top_quality_evidence = 0.0;
  double top_fused_confidence = 0.0;
  int top_observation_num = 0;
  int top_observation_cloud_sum = 0;
  double top_observability = 0.0;
  double top_distance = 0.0;
  double top_view_angle = 0.0;
  double top_mask_scale = 0.0;
  bool top_passes_threshold = false;
  int second_label = -1;
  double second_quality_evidence = 0.0;
  double second_fused_confidence = 0.0;
  int second_observation_num = 0;
  int second_observation_cloud_sum = 0;
  double second_observability = 0.0;
  double second_distance = 0.0;
  double second_view_angle = 0.0;
  double second_mask_scale = 0.0;
  bool second_passes_threshold = false;
  double top_second_abs_diff = 0.0;
  bool target_is_top_label = false;
};

struct VerificationCandidate {
  int object_id = -1;
  Vector2d position = Vector2d::Zero();
  double top_score = 0.0;
  double second_score = 0.0;
  double margin = 0.0;
  double target_mu_v = 1.0;
  double target_sigma_v = 0.35;
  int source_agent_id = -1;
  int verify_count = 0;
  bool verified = false;
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> object_cloud;
};

struct Viewpoint2D {
  Vector2d pos_;   ///< 2D position of viewpoint in world coordinates
  double yaw_;     ///< Heading angle in radians
  int visib_num_;  ///< Number of visible objects from this viewpoint
};

/**
 * @brief Object cluster with multi-modal information
 *
 * Represents a semantic object cluster combining 2D grid information,
 * 3D point clouds from multiple detections, confidence scores, and
 * geometric properties for robust object representation.
 */
struct ObjectCluster {
  /******* 2D Grid Information *******/
  int id_;                               ///< Unique cluster identifier
  vector<Vector2d> cells_;               ///< All 2D grid cells belonging to this cluster
  unordered_map<int, int> seen_counts_;  ///< Observation count per grid cell
  unordered_map<int, char> visited_;     ///< Visited flag per grid cell
  Vector2d average_;                     ///< Centroid position of all grid cells
  Vector2d box_min2d_, box_max2d_;       ///< 2D bounding box (min/max corners)
  Vector3d box_min3d_, box_max3d_;       ///< 3D bounding box from point clouds
  int max_seen_count_;                   ///< Maximum observation count across all cells
  vector<Vector2d> good_cells_;          ///< High-confidence cells (frequently observed)
  int best_label_;                       ///< Most confident semantic label
  int verification_count_ = 0;            ///< Number of cross-agent verification exports
  bool verification_pending_ = false;     ///< Waiting for verification before strict reach
  bool verification_verified_ = false;    ///< Verification margin has been satisfied
  int last_observer_agent_ = -1;          ///< Last agent that contributed target evidence

  /******* 3D Point Cloud Information *******/
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds_;  ///< Point clouds per semantic
                                                                    ///< class
  vector<double> confidence_scores_;    ///< Confidence scores per semantic class
  vector<double> observability_scores_;  ///< View-conditioned observability per semantic class
  vector<double> quality_evidence_scores_;  ///< Quality-aware semantic evidence per class
  vector<double> last_distances_;        ///< Latest observation distance per class
  vector<double> last_view_angles_;      ///< Latest observation angle per class
  vector<double> last_mask_scales_;      ///< Latest mask scale per class
  vector<int> observation_nums_;        ///< Number of observations per class
  vector<int> observation_cloud_sums_;  ///< Total point count per class

  /**
   * @brief Constructor to initialize multi-class storage
   * @param size Number of semantic classes to support (default: target + up to 5 confusions)
   */
  ObjectCluster(int size = 6)
    : clouds_(size)
    , confidence_scores_(size, 0.0)
    , observability_scores_(size, 0.0)
    , quality_evidence_scores_(size, 0.0)
    , last_distances_(size, 0.0)
    , last_view_angles_(size, 0.0)
    , last_mask_scales_(size, 0.0)
    , observation_nums_(size, 0)
    , observation_cloud_sums_(size, 0)
  {
  }
};

class ObjectMap2D {
public:
  ObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~ObjectMap2D() = default;

  void reset();  ///< Clear all object data for a new episode
  int searchSingleObjectCluster(const DetectedObject& detected_object);
  void inputObservationObjectsCloud(
      const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> observation_clouds,
      const double& itm_score);
  void setConfidenceThreshold(double val);
  bool getSemanticEvidenceSnapshot(int object_id, int label, SemanticEvidenceSnapshot& snapshot);
  void getSemanticEvidenceConfig(SemanticEvidenceSnapshot& snapshot) const;
  void getVerificationCandidates(std::vector<VerificationCandidate>& candidates);

  void getAllConfidenceObjectClouds(pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_clouds);
  void getTopConfidenceObjectCloud(
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& top_object_clouds,
      bool limited_confidence = true, bool extreme = false);
  void getObjects(
      vector<vector<Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels);
  void getObjectBoxes(vector<pair<Vector2d, Vector2d>>& boxes);
  void getObjectBoxes(vector<pair<Vector3d, Vector3d>>& boxes);
  void getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax);
  int getObjectGrid(const Eigen::Vector2d& pos);
  int getObjectGrid(const Eigen::Vector2i& id);
  int getObjectGrid(const int& adr);

  void publishObjectClouds();
  void wrapYaw(double& yaw);

  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> all_object_clouds_;
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> over_depth_object_cloud_;

private:
  double fusionConfidenceScore(
      int total_last, double c_last, int n_now, double c_now, int total_now, int sum);
  void updateQualityAwareEvidence(
      ObjectCluster& object, int label, const DetectedObject& detected_object);
  void ensureObjectLabelCapacity(ObjectCluster& object, int label);
  void updateObjectBestLabel(int obj_idx);
  void updateVerificationState(ObjectCluster& object);
  bool updateObject3DBounds(ObjectCluster& object, int label);
  Eigen::Vector4d getColor(const double& h, double alpha);

  bool haveOverlap(
      const Vector2d& min1, const Vector2d& max1, const Vector2d& min2, const Vector2d& max2);
  void createNewObjectCluster(
      const std::vector<Eigen::Vector2d>& cells, const DetectedObject& detected_object);
  void mergeCellsIntoObjectCluster(const int& object_id, const std::vector<Eigen::Vector2d>& cells,
      const DetectedObject& detected_object);

  vector<Eigen::Vector2i> fourNeighbors(const Eigen::Vector2i& idx);
  vector<Eigen::Vector2i> allNeighbors(const Eigen::Vector2i& idx);
  vector<Eigen::Vector2i> allGridsDistance(const Eigen::Vector2i& idx, const double& dist);
  bool isConfidenceObject(const ObjectCluster& obj);
  bool isNeighborUnknown(const Eigen::Vector2i& idx);
  bool isSatisfyObject(const Eigen::Vector2i& idx);
  bool isSatisfyObject(const Eigen::Vector2d& pos);
  bool isObjectClustered(const int& adr);
  bool isObjectClustered(const Eigen::Vector2d& pos);
  bool isObjectClustered(const Eigen::Vector2i& idx);
  void printFusionInfo(const ObjectCluster& obj, int label, const char* state);

  // Wrapper of sdf map
  int toAdr(const Eigen::Vector2d& pos);
  int toAdr(const Eigen::Vector2i& idx);
  bool knownFree(const Eigen::Vector2i& idx);
  bool inMap(const Eigen::Vector2i& idx);
  bool checkSafety(const Eigen::Vector2i& idx);
  bool checkSafety(const Eigen::Vector2d& pos);

  // ==================== Data Members ====================
  ros::Publisher object_cloud_pub_;  ///< Publisher for colored object visualization

  // Object storage and indexing
  vector<int> object_indexs_;      ///< Grid cell to object ID mapping
  vector<char> object_buffer_;     ///< Object occupancy grid buffer
  vector<ObjectCluster> objects_;  ///< Collection of all object clusters

  // Algorithm parameters
  bool use_observation_;     ///< Whether to use observation-based confidence reduction
  bool use_semantic_observability_;  ///< Whether Mission 2 score drives object selection
  bool is_vis_cloud_;        ///< Whether to publish visualization clouds
  int fusion_type_;          ///< Confidence fusion algorithm type (0=replace, 1=weighted, 2=max)
  int min_observation_num_;  ///< Minimum observations required for confidence
  double min_confidence_;    ///< Minimum confidence threshold for object acceptance
  double min_semantic_evidence_;  ///< Minimum quality-aware evidence threshold
  double verification_margin_threshold_;  ///< Ambiguous top-second margin for verification
  double verification_accept_margin_;  ///< Margin needed to release pending verification
  int verification_max_count_;  ///< Max verification exports for one object
  double lambda_d_;          ///< Distance decay coefficient
  double r0_;                ///< Mask sigmoid midpoint
  double mask_sigmoid_k_;    ///< Mask sigmoid slope
  double beta_;              ///< Observation saturation coefficient
  double resolution_;        ///< Grid resolution in meters
  double leaf_size_;         ///< Voxel size for point cloud downsampling

  // System integration
  SDFMap2D* sdf_map_;
  unique_ptr<RayCaster2D> raycaster_;
};

inline void ObjectMap2D::printFusionInfo(const ObjectCluster& obj, int label, const char* state)
{
  if (label < 0 || label >= (int)obj.confidence_scores_.size())
    return;

  // Use purple for high-confidence label 0, green for others
  if (label == 0 && obj.confidence_scores_[label] >= min_confidence_)
    ROS_WARN("%s%s id = %d label = %d confidence score = %.3lf %s", T_COLORS[5], state, obj.id_,
        label, obj.confidence_scores_[label], T_COLORS[0]);
  else
    ROS_WARN("%s%s id = %d label = %d confidence score = %.3lf %s", T_COLORS[2], state, obj.id_,
        label, obj.confidence_scores_[label], T_COLORS[0]);
}

inline bool ObjectMap2D::isSatisfyObject(const Eigen::Vector2i& idx)
{
  Vector2d pos;
  sdf_map_->indexToPos(idx, pos);
  return isSatisfyObject(pos);
}

inline bool ObjectMap2D::isSatisfyObject(const Eigen::Vector2d& pos)
{
  if (sdf_map_->isInMap(pos) && sdf_map_->getOccupancy(pos) == SDFMap2D::OCCUPIED)
    return true;
  return false;
}

inline bool ObjectMap2D::isObjectClustered(const int& adr)
{
  if (adr < 0 || adr >= (int)object_indexs_.size())
    return false;
  if (object_indexs_[adr] == -1)
    return false;
  return true;
}

inline bool ObjectMap2D::isObjectClustered(const Eigen::Vector2i& idx)
{
  if (!sdf_map_->isInMap(idx))
    return false;
  return isObjectClustered(toAdr(idx));
}

inline bool ObjectMap2D::isObjectClustered(const Eigen::Vector2d& pos)
{
  if (!sdf_map_->isInMap(pos))
    return false;
  Eigen::Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return isObjectClustered(idx);
}

inline bool ObjectMap2D::haveOverlap(
    const Vector2d& min1, const Vector2d& max1, const Vector2d& min2, const Vector2d& max2)
{
  // Check for separation along each axis
  Vector2d bmin, bmax;
  for (int i = 0; i < 2; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3)
      return false;
  }
  return true;
}

inline void ObjectMap2D::wrapYaw(double& yaw)
{
  while (yaw < -M_PI) yaw += 2 * M_PI;
  while (yaw > M_PI) yaw -= 2 * M_PI;
}

inline vector<Eigen::Vector2i> ObjectMap2D::fourNeighbors(const Eigen::Vector2i& idx)
{
  vector<Eigen::Vector2i> neighbors(4);
  neighbors[0] = idx + Eigen::Vector2i(-1, 0);
  neighbors[1] = idx + Eigen::Vector2i(1, 0);
  neighbors[2] = idx + Eigen::Vector2i(0, -1);
  neighbors[3] = idx + Eigen::Vector2i(0, 1);
  return neighbors;
}

inline vector<Eigen::Vector2i> ObjectMap2D::allNeighbors(const Eigen::Vector2i& idx)
{
  vector<Eigen::Vector2i> neighbors(8);
  int count = 0;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center cell
      neighbors[count++] = idx + Eigen::Vector2i(x, y);
    }
  }
  return neighbors;
}

inline vector<Eigen::Vector2i> ObjectMap2D::allGridsDistance(
    const Eigen::Vector2i& idx, const double& dist)
{
  vector<Eigen::Vector2i> grids;
  int cnt = ceil(dist / resolution_);  // Convert distance to grid cells

  for (int x = -cnt; x <= cnt; ++x) {
    for (int y = -cnt; y <= cnt; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center cell

      // Check if grid point is within circular distance
      Vector2d step_dist = Vector2d(x * resolution_, y * resolution_);
      if (step_dist.norm() <= dist) {
        Eigen::Vector2i grid_idx = idx + Eigen::Vector2i(x, y);
        grids.push_back(grid_idx);
      }
    }
  }
  return grids;
}

inline bool ObjectMap2D::isNeighborUnknown(const Eigen::Vector2i& idx)
{
  auto nbrs = fourNeighbors(idx);
  for (auto nbr : nbrs) {
    if (sdf_map_->getOccupancy(nbr) == SDFMap2D::UNKNOWN)
      return true;
  }
  return false;
}

inline int ObjectMap2D::toAdr(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return sdf_map_->toAddress(idx);
}

inline int ObjectMap2D::toAdr(const Eigen::Vector2i& idx)
{
  return sdf_map_->toAddress(idx);
}

inline bool ObjectMap2D::knownFree(const Eigen::Vector2i& idx)
{
  return sdf_map_->getOccupancy(idx) == SDFMap2D::FREE;
}

inline bool ObjectMap2D::inMap(const Eigen::Vector2i& idx)
{
  return sdf_map_->isInMap(idx);
}

inline int ObjectMap2D::getObjectGrid(const int& adr)
{
  if (adr < 0 || adr >= (int)object_buffer_.size())
    return -1;
  return int(object_buffer_[adr]);
}

inline int ObjectMap2D::getObjectGrid(const Eigen::Vector2i& id)
{
  if (!sdf_map_->isInMap(id))
    return -1;  // Invalid position
  return int(getObjectGrid(sdf_map_->toAddress(id)));
}

inline int ObjectMap2D::getObjectGrid(const Eigen::Vector2d& pos)
{
  Eigen::Vector2i id;
  sdf_map_->posToIndex(pos, id);
  return getObjectGrid(id);
}

inline Eigen::Vector4d ObjectMap2D::getColor(const double& h, double alpha)
{
  double h1 = h;
  if (h1 < 0.0 || h1 > 1.0) {
    h1 = std::fmod(std::max(0.0, h1), 1.0);
  }

  double lambda;
  Eigen::Vector4d color1, color2;

  // HSV color wheel interpolation with 6 segments
  if (h1 >= -1e-4 && h1 < 1.0 / 6) {
    lambda = (h1 - 0.0) * 6;
    color1 = Eigen::Vector4d(1, 0, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 1, 1);
  }
  else if (h1 >= 1.0 / 6 && h1 < 2.0 / 6) {
    lambda = (h1 - 1.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 0, 1, 1);
  }
  else if (h1 >= 2.0 / 6 && h1 < 3.0 / 6) {
    lambda = (h1 - 2.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 1, 1);
  }
  else if (h1 >= 3.0 / 6 && h1 < 4.0 / 6) {
    lambda = (h1 - 3.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 0, 1);
  }
  else if (h1 >= 4.0 / 6 && h1 < 5.0 / 6) {
    lambda = (h1 - 4.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 1, 0, 1);
  }
  else if (h1 >= 5.0 / 6 && h1 <= 1.0 + 1e-4) {
    lambda = (h1 - 5.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 0, 1);
  }

  Eigen::Vector4d fcolor = (1 - lambda) * color1 + lambda * color2;
  fcolor(3) = alpha;

  return fcolor;
}

inline void ObjectMap2D::publishObjectClouds()
{
  // Create colored point cloud container
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_colored_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // Process each detected object
  for (const auto& object : objects_) {
    if (object.best_label_ == -1)
      continue;  // Skip objects without valid classification

    // Get the best-confidence point cloud for this object
    if (object.best_label_ < 0 || object.best_label_ >= (int)object.clouds_.size())
      continue;
    const auto& cloud = object.clouds_[object.best_label_];
    if (!cloud)
      continue;

    // Color and transform each point
    for (const auto& point : cloud->points) {
      pcl::PointXYZRGB colored_point;
      colored_point.x = point.x;
      colored_point.y = point.y;
      colored_point.z = point.z + 1.0;  // Elevate for better visibility

      // Generate class-specific color with soft blending
      Eigen::Vector4d col = getColor(object.best_label_ / 5.0, 1.0);
      double blend_factor = 0.5;  // Softening factor for pastel colors
      uint8_t r = col(0) * 255 * blend_factor + 255 * (1.0 - blend_factor);
      uint8_t g = col(1) * 255 * blend_factor + 255 * (1.0 - blend_factor);
      uint8_t b = col(2) * 255 * blend_factor + 255 * (1.0 - blend_factor);
      colored_point.r = r;
      colored_point.g = g;
      colored_point.b = b;

      combined_colored_cloud->points.push_back(colored_point);
    }
  }

  if (combined_colored_cloud->points.empty())
    return;  // No objects to visualize

  // Configure point cloud metadata
  combined_colored_cloud->width = combined_colored_cloud->points.size();
  combined_colored_cloud->height = 1;
  combined_colored_cloud->is_dense = true;

  // Publish the combined visualization
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(*combined_colored_cloud, output);
  output.header.frame_id = "world";
  output.header.stamp = ros::Time::now();
  object_cloud_pub_.publish(output);
}
}  // namespace apexnav_planner

#endif
