#ifndef _EXPLORATION_FSM_REAL_H_
#define _EXPLORATION_FSM_REAL_H_

// Third-party libraries
#include <Eigen/Eigen>

// Standard C++ libraries
#include <memory>
#include <string>
#include <vector>

// ROS core
#include <ros/ros.h>

// ROS message types
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <trajectory_manager/PolyTraj.h>
#include <exploration_manager/robust_navigation_policy.h>

namespace apexnav_planner {

// Forward declarations
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;
struct LocalTrajectory;

namespace FSMConstantsReal {
// Timers (s)
constexpr double EXEC_TIMER_DURATION = 0.01;
constexpr double FRONTIER_TIMER_DURATION = 0.25;
constexpr double REPLAN_CHECK_DURATION = 0.1;  // Check if replan needed

// Trajectory execution
constexpr double TRAJECTORY_EXECUTION_TIMEOUT = 10.0;  // Max time to execute trajectory
constexpr double GOAL_REACH_THRESHOLD = 0.15;         // Distance to consider goal reached
constexpr double REPLAN_DISTANCE_THRESHOLD = 0.5;     // Trigger replan if deviate too much

// Distances (m)
constexpr double STUCKING_DISTANCE = 0.05;
constexpr double REACH_DISTANCE = 0.20;
constexpr double SOFT_REACH_DISTANCE = 0.45;
constexpr double LOCAL_DISTANCE = 0.80;
constexpr double FORWARD_DISTANCE = 0.15;
constexpr double FORCE_DORMANT_DISTANCE = 0.35;
constexpr double MIN_SAFE_DISTANCE = 0.15;

// Counters / thresholds
constexpr int MAX_STUCKING_COUNT = 25;
constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;
constexpr int MAX_REPLAN_FAILURES = 3;  // Max consecutive replan failures

// Cost weights
constexpr double TARGET_WEIGHT = 150.0;
constexpr double TARGET_CLOSE_WEIGHT_1 = 2000.0;
constexpr double TARGET_CLOSE_WEIGHT_2 = 200.0;
constexpr double SAFETY_WEIGHT = 1.0;
constexpr double SAMPLE_NUM = 10.0;

// Visualization / robot marker
constexpr double VIS_SCALE_FACTOR = 1.8;
constexpr double ROBOT_HEIGHT = 0.15;
constexpr double ROBOT_RADIUS = 0.18;
}  // namespace FSMConstantsReal

class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

// Real-world FSM states (using class enum to avoid conflicts)
namespace RealFSM {
  enum class State {
    INIT,
    WAIT_TRIGGER,
    PLAN_TRAJ,           // Plan continuous trajectory
    EXEC_TRAJ,           // Executing trajectory
    REPLAN,              // Replanning during execution
    FINISH
  };

  enum class Result {
    EXPLORE,
    SEARCH_OBJECT,
    STUCKING,
    NO_FRONTIER,
    REACH_OBJECT,
    MANUAL_STOP
  };
}

// Trajectory planner return codes
enum class TrajPlannerResult {
  RETRYABLE_FAILED = 0,
  SUCCESS = 1,
  MISSION_SUCCEEDED = 2,
  MISSION_FAILED = 3,
};

// Real-world exploration FSM for continuous trajectory execution
class ExplorationFSMReal {
private:
  /* Planning Utils */
  ros::NodeHandle nh_;
  std::shared_ptr<FastPlannerManager> planner_manager_;
  std::shared_ptr<ExplorationManager> expl_manager_;
  std::shared_ptr<PlanningVisualization> visualization_;

  std::shared_ptr<FSMParam> fp_;
  std::shared_ptr<FSMData> fd_;
  RealFSM::State state_;
  std::string world_frame_;
  std::string base_frame_;
  double trigger_max_age_;
  double odometry_max_age_;
  double input_max_future_;
  double odometry_quaternion_norm_tolerance_;
  double first_trajectory_progress_limit_;
  ros::Time last_trigger_source_stamp_;
  ros::Time last_odometry_source_stamp_;
  ros::Time next_plan_attempt_time_;
  ros::Time tracking_error_since_;
  ros::Time trajectory_execution_start_;
  Eigen::Vector2d trajectory_start_odom_;
  double trajectory_progress_;
  ros::Time last_trajectory_progress_;
  bool trajectory_progress_started_;
  bool require_replan_settle_;
  ReplanSettleGate replan_settle_gate_;
  bool have_locked_target_;
  Eigen::Vector2d locked_target_;
  int locked_target_result_;
  int locked_target_failures_;
  ros::Time locked_target_since_;
  ros::Time locked_target_first_failure_since_;
  double target_failure_min_hysteresis_;
  bool navigation_enabled_ = false;

  /* ROS Utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, frontier_timer_, safety_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, confidence_threshold_sub_, cancel_sub_;
  ros::Subscriber trajectory_progress_sub_;
  ros::Subscriber navigation_enabled_sub_;
  ros::Subscriber traj_finish_sub_;  // TODO: Subscribe to trajectory execution status
  
  ros::Publisher ros_state_pub_, expl_state_pub_, expl_result_pub_;
  ros::Publisher robot_marker_pub_;
  
  // Real-world specific: trajectory control publishers
  ros::Publisher poly_traj_pub_;   // Publish polynomial trajectory
  ros::Publisher stop_pub_;        // Emergency stop signal
  
  /* Trajectory execution status */
  // Trajectory state is tracked in fd_->static_state_

  /* Exploration Planner */
  TrajPlannerResult callTrajectoryPlanner();
  bool enforceTrajectoryLimits(LocalTrajectory& local_traj);
  bool validateTrajectoryStructure(
      const LocalTrajectory& local_traj, const char* stage, double* total_duration) const;
  bool validateFinalTrajectory(const LocalTrajectory& local_traj) const;
  void polyTraj2ROSMsg(const LocalTrajectory& local_traj, trajectory_manager::PolyTraj& poly_msg);
  bool selectLocalTarget(const Eigen::Vector2d& current_pos, const std::vector<Eigen::Vector2d>& path,
      const double& local_distance, Eigen::Vector2d& target_pos, double& target_yaw);
  
  // Safety and stuck detection
  void emergencyStop();
  bool checkNeedReplan();
  bool checkStuckCondition();
  double computePathCost(const std::vector<Eigen::Vector2d>& path);

  /* Helper functions */
  bool updateFrontierAndObject();
  void recordLockedTargetFailure();
  void resetLockedTargetFailures();
  void transitState(RealFSM::State new_state, std::string pos_call);
  void wrapAngle(double& angle);
  void publishRobotMarker();
  void visualize();
  void clearVisMarker();

  /* ROS callbacks */
  void FSMCallback(const ros::TimerEvent& e);
  void safetyCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg);
  void cancelCallback(const std_msgs::EmptyConstPtr& msg);
  void trajectoryFinishCallback(const std_msgs::EmptyConstPtr& msg);  // TODO: Define proper msg type
  void trajectoryProgressCallback(const std_msgs::Float64ConstPtr& msg);
  void navigationEnabledCallback(const std_msgs::BoolConstPtr& msg);

public:
  ExplorationFSMReal() = default;
  ~ExplorationFSMReal() = default;

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void ExplorationFSMReal::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}

}  // namespace apexnav_planner

#endif
