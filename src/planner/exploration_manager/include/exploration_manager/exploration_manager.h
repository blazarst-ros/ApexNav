#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

// Third-party libraries
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Standard C++ libraries
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <exploration_manager/exploration_data.h>

// ROS core
#include <ros/ros.h>

// Plan environment
#include <plan_env/frontier_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>

// Path searching
#include <path_searching/astar2d.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
class SDFMap2D;
class FrontierMap2D;
class Gcopter;
class KinoAstar;
struct ExplorationParam;

struct SemanticFrontier {
  Vector2d position;      ///< 2D position of the frontier
  double semantic_value;  ///< Semantic value at the frontier location
  double path_length;     ///< Path length to reach this frontier
  vector<Vector2d> path;  ///< Complete path to the frontier

  bool operator<(const SemanticFrontier& other) const
  {
    if (fabs(semantic_value - other.semantic_value) < 1e-4) {
      // If semantic values are equal, sort by path length (ascending)
      return path_length < other.path_length;
    }
    // Otherwise, sort by semantic value (descending)
    return semantic_value > other.semantic_value;
  }
};

// A planning mode is more specific than the final result sent to Habitat:
// only equal modes are eligible for a shared assignment.
enum class NavigationMode {
  NONE,
  SEARCH_BEST_OBJECT,
  SEARCH_OVER_DEPTH_OBJECT,
  SEARCH_SUSPICIOUS_OBJECT,
  SEMANTIC_FRONTIER,
  GEOMETRIC_FRONTIER,
  DORMANT_FRONTIER,
  SEARCH_EXTREME
};

struct JointAssignment {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  NavigationMode mode = NavigationMode::NONE;
  std::array<Vector2d, NUM_AGENTS> next_positions;
  std::array<vector<Vector2d>, NUM_AGENTS> next_paths;
};

class ExplorationManager {
public:
  ExplorationManager() = default;
  ~ExplorationManager();  // Explicit destructor declaration for shared_ptr with forward declaration

  void initialize(ros::NodeHandle& nh);
  void resetEpisodeState();

  int planNextBestPoint(const Vector3d& pos, const double& yaw, int agent_idx,
      Eigen::Vector2d& out_next_pos, std::vector<Eigen::Vector2d>& out_next_best_path);
  bool planTrajectory(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl);
  void getSortedSemanticFrontiers(const Vector2d& cur_pos, const vector<Vector2d>& frontiers,
      vector<SemanticFrontier>& sem_frontiers);
  void calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers, double& std_dev,
      double& max_to_mean, double& mean, bool if_print = false);
  NavigationMode evaluateNavigationMode(const Vector3d& pos, int agent_idx);
  bool planJointModeTargets(const std::array<Vector3d, NUM_AGENTS>& agent_positions,
      NavigationMode mode, JointAssignment& assignment);

  shared_ptr<ExplorationData> ed_;            ///< Exploration data container
  shared_ptr<ExplorationParam> ep_;           ///< Exploration parameters
  unique_ptr<Astar2D> path_finder_;           ///< A* path finding algorithm
  shared_ptr<FrontierMap2D> frontier_map2d_;  ///< 2D frontier map
  shared_ptr<ObjectMap2D> object_map2d_;      ///< 2D object map
  shared_ptr<SDFMap2D> sdf_map_;              ///< Signed distance field map
  shared_ptr<Gcopter> gcopter_;               ///< Trajectory optimizer (for real-world)
  shared_ptr<KinoAstar> kinoastar_;           ///< Kinodynamic A* planner (for real-world)

  typedef shared_ptr<ExplorationManager> Ptr;

private:
  // Exploration Policy
  void chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx);
  void findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx);
  void findHighestSemanticsFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx);
  void hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path, int agent_idx);
  void findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path, int agent_idx);

  // Path Search Utils
  bool searchObjectPath(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchObjectPathExtreme(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchFrontierPath(const Vector2d& start, const Vector2d& end, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path);
  void shortenPath(vector<Vector2d>& path);

  // Helper functions for object path searching
  Vector2d findNearestObjectPoint(
      const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud);
  bool trySearchObjectPathWithDistance(const Vector2d& start2d, const Vector2d& object_pose,
      double distance, double max_search_time, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path, const std::string& debug_msg);

  // TSP Optimization Methods
  void computeATSPTour(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices);
  void computeATSPCostMatrix(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& cost_matrix);
  double computePathCost(const Vector2d& pos1, const Vector2d& pos2);
  vector<Vector2i> allNeighbors(const Eigen::Vector2i& idx, int grid_radius);
  double getFrontierSemanticValue(const Vector2d& frontier);
  int findFrontierIdByPosition(const Vector2d& frontier, bool dormant = false);
  void setStrategyInfo(int agent_idx, const std::string& mode, const std::string& target_type,
      int target_id, double semantic_score, const vector<Vector2d>& path,
      const Vector2d& target_pos);

  struct JointCandidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int target_id = -1;
    std::string target_type;
    double semantic_score = 0.0;
    std::array<Vector2d, NUM_AGENTS> target_positions;
    std::array<vector<Vector2d>, NUM_AGENTS> initial_paths;
    std::array<double, NUM_AGENTS> initial_costs;
  };

  bool collectJointCandidates(const std::array<Vector3d, NUM_AGENTS>& agent_positions,
      NavigationMode mode, vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>>& candidates);
  bool computeJointMtspCostMatrix(
      const vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>>& candidates,
      std::array<Eigen::MatrixXd, NUM_AGENTS>& transition_costs);
  bool solveTwoStartMinmax(const vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>>& candidates,
      const std::array<Eigen::MatrixXd, NUM_AGENTS>& transition_costs,
      std::array<vector<int>, NUM_AGENTS>& routes) const;
  static const char* navigationModeName(NavigationMode mode);

  ros::ServiceClient tsp_client_;         ///< ROS service client for TSP solver
  unique_ptr<RayCaster2D> ray_caster2d_;  ///< Ray casting for collision checking
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> last_over_depth_object_cloud_;
};

inline bool ExplorationManager::searchFrontierPath(const Vector2d& start, const Vector2d& end,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start, end, 0.25, 0.01) == Astar2D::REACH_END) {
    refined_pos = end;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline bool ExplorationManager::searchObjectPathExtreme(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error finding nearest point

  Vector2d start2d = Vector2d(start(0), start(1));
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, 0.25, 0.2, Astar2D::SAFETY_MODE::EXTREME) ==
      Astar2D::REACH_END) {
    refined_pos = object_pose;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline void ExplorationManager::shortenPath(vector<Vector2d>& path)
{
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  // Shorten the path by keeping only critical intermediate points
  const double dist_thresh = 3.0;  // Minimum distance threshold for waypoint retention
  vector<Vector2d> short_tour = { path.front() };

  for (int i = 1; i < (int)path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints only when necessary to avoid collision
      ray_caster2d_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector2i idx;
      while (ray_caster2d_->nextId(idx) && ros::ok()) {
        if (sdf_map_->getInflateOccupancy(idx) == 1 ||
            sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  // Always include the final destination
  if ((path.back() - short_tour.back()).norm() > 1e-3)
    short_tour.push_back(path.back());

  // Ensure minimum path complexity (at least three points)
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));

  path = short_tour;
}

inline vector<Eigen::Vector2i> ExplorationManager::allNeighbors(
    const Eigen::Vector2i& idx, int grid_radius)
{
  vector<Eigen::Vector2i> neighbors;

  for (int x = -grid_radius; x <= grid_radius; ++x) {
    for (int y = -grid_radius; y <= grid_radius; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center point
      Eigen::Vector2i offset(x, y);
      neighbors.push_back(idx + offset);
    }
  }
  return neighbors;
}

}  // namespace apexnav_planner

#endif
