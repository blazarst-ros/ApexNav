#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <vector>
#include <trajectory_manager/optimizer.h>

// Undefine uint macro from optimizer.h to avoid conflict with OpenCV
#ifdef uint
#undef uint
#endif

namespace apexnav_planner {

static constexpr int NUM_AGENTS = 2;

enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };
enum EXPL_RESULT {
  EXPLORATION,               ///< Normal exploration mode
  SEARCH_BEST_OBJECT,        ///< Found high-confidence object
  SEARCH_OVER_DEPTH_OBJECT,  ///< Searching over-depth object
  SEARCH_SUSPICIOUS_OBJECT,  ///< Investigating suspicious object
  NO_PASSABLE_FRONTIER,      ///< No reachable frontiers available
  NO_COVERABLE_FRONTIER,     ///< No coverable frontiers found
  SEARCH_EXTREME             ///< Extreme search mode activated
};

struct AgentFSMData {
  AgentFSMData()
  {
    have_odom_ = false;
    have_finished_ = false;
    trigger_ = false;
    odom_pos_ = Eigen::Vector3d::Zero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_yaw_ = 0.0;
    start_pt_ = Eigen::Vector3d::Zero();
    start_yaw_ = 0.0;
    start_yaw_rate_ = 0.0;
    last_start_pos_ = Eigen::Vector3d(-100, -100, -100);
    last_next_pos_ = Eigen::Vector2d(-100, -100);
    newest_action_ = -1;
    init_action_count_ = 0;
    stucking_action_count_ = 0;
    stucking_next_pos_count_ = 0;
    traveled_path_.clear();

    final_result_ = -1;
    expl_result_ = EXPL_RESULT::EXPLORATION;
    replan_flag_ = true;
    dormant_frontier_flag_ = false;
    escape_stucking_flag_ = false;
    escape_stucking_count_ = 0;
    stucking_points_.clear();

    local_pos_ = Eigen::Vector2d(0, 0);
    odom_vel_ = Eigen::Vector3d::Zero();
    start_vel_ = Eigen::Vector3d::Zero();
    odom_omega_ = Eigen::Vector3d::Zero();
  }
  bool have_odom_, have_finished_, trigger_;
  Eigen::Vector3d odom_pos_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;
  Eigen::Vector3d start_pt_;
  double start_yaw_;
  double start_yaw_rate_;
  Eigen::Vector3d last_start_pos_;
  Eigen::Vector2d last_next_pos_;
  int newest_action_;
  int init_action_count_;
  int stucking_action_count_;
  int stucking_next_pos_count_;
  int final_result_;
  int expl_result_;
  bool replan_flag_, dormant_frontier_flag_;
  bool escape_stucking_flag_;
  int escape_stucking_count_;
  Eigen::Vector2d escape_stucking_pos_;
  double escape_stucking_yaw_;
  std::vector<Eigen::Vector3d> stucking_points_;
  Eigen::Vector2d local_pos_;
  Eigen::Vector3d odom_vel_;
  Eigen::Vector3d start_vel_;
  Eigen::Vector3d odom_omega_;
  std::vector<Eigen::Vector2d> traveled_path_;
  LocalTrajectory newest_traj_;

  // Per-agent planning output (decoupled from shared ExplorationData)
  Eigen::Vector2d planned_next_pos_;
  std::vector<Eigen::Vector2d> planned_next_best_path_;
};

struct FSMData {
  FSMData()
  {
    agent_.assign(NUM_AGENTS, AgentFSMData());
    trigger_ = false;
    have_confidence_ = false;
    static_state_ = true;
  }
  // FSM data
  bool trigger_, have_confidence_;
  bool static_state_;
  std::vector<AgentFSMData> agent_;  // per-agent data
};

struct FSMParam {
  FSMParam()
  {
    vis_scale_ = 0.1;
    replan_time_ = 0.2;
    replan_traj_end_threshold_ = 1.0;
    replan_frontier_change_delay_ = 0.5;
    replan_timeout_ = 2.0;

    const double step_length = 0.25;
    const double angle_increment = M_PI / 6;
    action_steps_.clear();
    for (int i = 0; i < 12; ++i) {
      double angle = i * angle_increment;
      Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
      action_steps_.push_back(step);
    }
  }
  double vis_scale_;
  std::vector<Eigen::Vector2d> action_steps_;
  // replan timing parameters (loaded from ros params in ExplorationFSM::init)
  double replan_time_;
  double replan_traj_end_threshold_;
  double replan_frontier_change_delay_;
  double replan_timeout_;
};

struct ExplorationData {
  struct NavigationStrategyInfo {
    NavigationStrategyInfo()
    {
      agent_id = -1;
      mode = "UNKNOWN";
      target_type = "NONE";
      target_id = -1;
      semantic_score = -1.0;
      path_length = -1.0;
      target_pos = Eigen::Vector2d(0, 0);
    }

    int agent_id;
    std::string mode;
    std::string target_type;
    int target_id;
    double semantic_score;
    double path_length;
    Eigen::Vector2d target_pos;
  };

  ExplorationData()
  {
    frontiers_.clear();
    frontier_averages_.clear();
    dormant_frontiers_.clear();
    dormant_frontier_averages_.clear();
    objects_.clear();
    object_averages_.clear();
    object_labels_.clear();
    next_pos_ = Eigen::Vector2d(0, 0);
    next_best_path_.clear();
    tsp_tour_.clear();
    strategy_infos_.assign(NUM_AGENTS, NavigationStrategyInfo());
  }
  std::vector<std::vector<Eigen::Vector2d>> frontiers_, dormant_frontiers_;
  std::vector<Eigen::Vector2d> frontier_averages_, dormant_frontier_averages_;
  std::vector<std::vector<Eigen::Vector2d>> objects_;
  std::vector<Eigen::Vector2d> object_averages_;
  std::vector<int> object_labels_;
  Eigen::Vector2d next_pos_;
  Eigen::Vector2d next_local_pos_;  // Local target position along path
  std::vector<Eigen::Vector2d> next_best_path_;
  std::vector<Eigen::Vector2d> tsp_tour_;
  std::vector<NavigationStrategyInfo> strategy_infos_;
};

struct ExplorationParam {
  enum POLICY_MODE { DISTANCE, SEMANTIC, HYBRID, TSP_DIST };
  // params
  int policy_mode_;
  double sigma_threshold_, max_to_mean_threshold_, max_to_mean_percentage_;
  std::string tsp_dir_;
};

}  // namespace apexnav_planner

#endif
