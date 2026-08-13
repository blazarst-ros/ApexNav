
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <exploration_manager/fsm_policy.h>
#include <vis_utils/planning_visualization.h>
#include <std_msgs/Int32MultiArray.h>
#include <boost/bind/bind.hpp>
#include <algorithm>
#include <sstream>

namespace apexnav_planner {
void ExplorationFSM::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.resize(NUM_AGENTS);
  for (int i = 0; i < NUM_AGENTS; ++i)
    visualization_[i].reset(new PlanningVisualization(nh, i));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;

  for (int i = 0; i < NUM_AGENTS; ++i)
    state_[i] = ROS_STATE::INIT;

  /* ROS Timer */
  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstants::EXEC_TIMER_DURATION), &ExplorationFSM::FSMCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstants::FRONTIER_TIMER_DURATION),
      &ExplorationFSM::frontierCallback, this);

  /* ROS Subscriber */
  trigger_sub_ = nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSM::triggerCallback, this);
  for (int i = 0; i < NUM_AGENTS; ++i) {
    std::string odom_topic = "/habitat/agent_" + std::to_string(i) + "/odom";
    odom_sub_[i] = nh.subscribe<nav_msgs::Odometry>(
        odom_topic, 30,
        boost::bind(&ExplorationFSM::odometryCallback, this, boost::placeholders::_1, i));
  }
  habitat_state_sub_ =
      nh.subscribe("/habitat/state", 30, &ExplorationFSM::habitatStateCallback, this);
  confidence_threshold_sub_ = node_.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSM::confidenceThresholdCallback, this);

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  ros_state_all_pub_ = nh.advertise<std_msgs::Int32MultiArray>("/ros/state_all", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  expl_result_all_pub_ = nh.advertise<std_msgs::Int32MultiArray>("/ros/expl_result_all", 10);
  final_result_all_pub_ =
      nh.advertise<std_msgs::Int32MultiArray>("/ros/final_result_all", 10);
  reach_claim_pub_ = nh.advertise<std_msgs::Int32MultiArray>("/ros/reach_claim", 10);
  for (int i = 0; i < NUM_AGENTS; ++i) {
    action_pub_[i] = nh.advertise<std_msgs::Int32>(
        "/habitat/plan_action_agent_" + std::to_string(i), 10);
    expl_result_agent_pub_[i] = nh.advertise<std_msgs::Int32>(
        "/ros/agent_" + std::to_string(i) + "/expl_result", 10);
    exploration_strategy_pub_[i] = nh.advertise<std_msgs::String>(
        "/ros/agent_" + std::to_string(i) + "/exploration_strategy", 10);
    robot_marker_pub_[i] = nh.advertise<visualization_msgs::Marker>(
        "/robot_agent_" + std::to_string(i), 10);
  }
}

// FSM between ROS and Habitat for action planning and execution (round-robin over agents)
void ExplorationFSM::FSMCallback(const ros::TimerEvent& e)
{
  exec_timer_.stop();
  std::lock_guard<std::mutex> lock(data_mutex_);

  // A joint assignment is allowed only when both agents enter the same planning
  // phase together.  Otherwise each agent continues through its independent
  // planner and no frontier Claim is used as a substitute for coordination.
  planAgentsForCycle();

  for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
    auto& ad = fd_->agent_[agent_idx];

    switch (state_[agent_idx]) {
      case ROS_STATE::INIT: {
        // Wait for odometry and target confidence threshold
        if (!ad.have_odom_ || !fd_->have_confidence_) {
          ROS_WARN_THROTTLE(
              1.0, "Agent %d: No odom || No target confidence threshold.", agent_idx);
          continue;
        }
        // Go to WAIT_TRIGGER when prerequisites are ready
        transitState(agent_idx, ROS_STATE::WAIT_TRIGGER, "FSM");
        break;
      }

      case ROS_STATE::WAIT_TRIGGER: {
        if (!ad.trigger_) {
          ROS_WARN_THROTTLE(1.0, "Agent %d: Wait for trigger.", agent_idx);
        }
        break;
      }

      case ROS_STATE::FINISH: {
        ROS_WARN_THROTTLE(1.0, "Agent %d: Waiting in reach-claim team terminal state.", agent_idx);
        break;
      }

      case ROS_STATE::FINISH_FAILURE: {
        if (!ad.have_finished_) {
          ad.have_finished_ = true;
          if (agent_idx < static_cast<int>(expl_manager_->ed_->strategy_infos_.size())) {
            auto& info = expl_manager_->ed_->strategy_infos_[agent_idx];
            info.agent_id = agent_idx;
            info.mode = "FINISH_FAILURE";
            info.target_type = "NONE";
            info.target_id = -1;
            info.path_length = -1.0;
          }
          std_msgs::Int32 action_msg;
          action_msg.data = ACTION::STOP;
          action_pub_[agent_idx].publish(action_msg);
          publishExplorationStrategy(agent_idx);
          ROS_WARN("Agent %d failed exploration (final_result=%d); other agents continue.",
              agent_idx, ad.final_result_);
        }
        break;
      }

      case ROS_STATE::PLAN_ACTION: {
        if (ad.init_action_count_ < 1 + 12 + 1 + 12) {
          if (ad.init_action_count_ < 1)
            ad.newest_action_ = ACTION::TURN_DOWN;
          else if (ad.init_action_count_ < 1 + 12)
            ad.newest_action_ = ACTION::TURN_LEFT;
          else if (ad.init_action_count_ < 1 + 12 + 1)
            ad.newest_action_ = ACTION::TURN_UP;
          else
            ad.newest_action_ = ACTION::TURN_LEFT;
          ROS_WARN("Agent %d Init Mode Process -----> (%d/26)", agent_idx, ad.init_action_count_);
          ad.init_action_count_++;
          transitState(agent_idx, ROS_STATE::PUB_ACTION, "FSM");
          updateFrontierAndObject();
        }
        else {
          // Main planning phase
          ad.start_pt_ = ad.odom_pos_;
          ad.start_yaw_ = ad.odom_yaw_;

          auto t1 = ros::Time::now();
          ad.final_result_ = callActionPlanner(agent_idx);
          double call_action_planner_time = (ros::Time::now() - t1).toSec();
          ROS_INFO_THROTTLE(
              10.0, "[Agent %d] Planning process time = %.3f s", agent_idx, call_action_planner_time);

          std_msgs::Int32 expl_state_msg;
          expl_state_msg.data = ad.final_result_;
          expl_state_pub_.publish(expl_state_msg);
          if (ad.final_result_ == FINAL_RESULT::REACH_OBJECT) {
            std_msgs::Int32MultiArray reach_claim_msg;
            reach_claim_msg.data = {agent_idx, FINAL_RESULT::REACH_OBJECT};
            reach_claim_pub_.publish(reach_claim_msg);
            ROS_WARN("Agent %d reached an object candidate; broadcasting STOP to all agents.",
                agent_idx);
            for (int stop_idx = 0; stop_idx < NUM_AGENTS; ++stop_idx) {
              std_msgs::Int32 action_msg;
              action_msg.data = ACTION::STOP;
              action_pub_[stop_idx].publish(action_msg);
              fd_->agent_[stop_idx].have_finished_ = true;
              transitState(stop_idx, ROS_STATE::FINISH, "Reach Object");
            }
          }
          else if (ad.final_result_ == FINAL_RESULT::EXPLORE ||
              ad.final_result_ == FINAL_RESULT::SEARCH_OBJECT)
            transitState(agent_idx, ROS_STATE::PUB_ACTION, "FSM");
          else
            transitState(agent_idx, ROS_STATE::FINISH_FAILURE, "Planner Failure");
        }
        visualize();
        break;
      }

      case ROS_STATE::PUB_ACTION: {
        std_msgs::Int32 action_msg;
        action_msg.data = ad.newest_action_;
        action_pub_[agent_idx].publish(action_msg);
        transitState(agent_idx, ROS_STATE::WAIT_ACTION_FINISH, "FSM");
        break;
      }

      case ROS_STATE::WAIT_ACTION_FINISH: {
        break;
      }
    }
  }
  publishPlannerState();
  exec_timer_.start();
}

void ExplorationFSM::publishPlannerState()
{
  std_msgs::Int32MultiArray state_all_msg;
  std_msgs::Int32MultiArray final_result_all_msg;
  state_all_msg.data.resize(NUM_AGENTS);
  final_result_all_msg.data.resize(NUM_AGENTS);
  for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
    state_all_msg.data[agent_idx] = state_[agent_idx];
    final_result_all_msg.data[agent_idx] = fd_->agent_[agent_idx].final_result_;
  }

  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = state_[0];
  ros_state_pub_.publish(ros_state_msg);
  ros_state_all_pub_.publish(state_all_msg);
  final_result_all_pub_.publish(final_result_all_msg);
}

void ExplorationFSM::publishExplorationResults()
{
  std_msgs::Int32MultiArray expl_result_all_msg;
  expl_result_all_msg.data.resize(NUM_AGENTS);
  for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
    expl_result_all_msg.data[agent_idx] = fd_->agent_[agent_idx].expl_result_;
    std_msgs::Int32 expl_result_agent_msg;
    expl_result_agent_msg.data = fd_->agent_[agent_idx].expl_result_;
    expl_result_agent_pub_[agent_idx].publish(expl_result_agent_msg);
    publishExplorationStrategy(agent_idx);
  }

  expl_result_all_pub_.publish(expl_result_all_msg);
}

void ExplorationFSM::publishExplorationStrategy(int agent_idx)
{
  if (agent_idx < 0 || agent_idx >= NUM_AGENTS)
    return;
  const auto& infos = expl_manager_->ed_->strategy_infos_;
  if (agent_idx >= static_cast<int>(infos.size()))
    return;

  const auto& info = infos[agent_idx];
  std_msgs::String msg;
  std::ostringstream ss;
  ss << "{"
     << "\"agent_id\":" << info.agent_id << ","
     << "\"mode\":\"" << info.mode << "\","
     << "\"target_type\":\"" << info.target_type << "\","
     << "\"target_id\":" << info.target_id << ","
     << "\"semantic_score\":" << info.semantic_score << ","
     << "\"path_length\":" << info.path_length << ","
     << "\"target_pos\":[" << info.target_pos(0) << "," << info.target_pos(1) << "]"
     << "}";
  msg.data = ss.str();
  exploration_strategy_pub_[agent_idx].publish(msg);
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
bool ExplorationFSM::planAgentsForCycle()
{
  std::array<Vector3d, NUM_AGENTS> agent_positions;
  for (int agent = 0; agent < NUM_AGENTS; ++agent) {
    const auto& ad = fd_->agent_[agent];
    if (state_[agent] != ROS_STATE::PLAN_ACTION || ad.init_action_count_ < 26 || !ad.have_odom_)
      return false;
    agent_positions[agent] = ad.odom_pos_;
  }

  const NavigationMode mode0 = expl_manager_->evaluateNavigationMode(agent_positions[0], 0);
  const NavigationMode mode1 = expl_manager_->evaluateNavigationMode(agent_positions[1], 1);
  if (mode0 != mode1 || mode0 == NavigationMode::NONE)
    return false;  // Different modes deliberately remain distributed and independent.

  JointAssignment assignment;
  if (!expl_manager_->planJointModeTargets(agent_positions, mode0, assignment) || !assignment.valid)
    return false;

  const int joint_result = (mode0 == NavigationMode::SEMANTIC_FRONTIER ||
      mode0 == NavigationMode::GEOMETRIC_FRONTIER || mode0 == NavigationMode::DORMANT_FRONTIER)
      ? EXPL_RESULT::EXPLORATION
      : (mode0 == NavigationMode::SEARCH_BEST_OBJECT ? EXPL_RESULT::SEARCH_BEST_OBJECT
         : mode0 == NavigationMode::SEARCH_OVER_DEPTH_OBJECT ? EXPL_RESULT::SEARCH_OVER_DEPTH_OBJECT
         : mode0 == NavigationMode::SEARCH_SUSPICIOUS_OBJECT ? EXPL_RESULT::SEARCH_SUSPICIOUS_OBJECT
         : EXPL_RESULT::SEARCH_EXTREME);
  for (int agent = 0; agent < NUM_AGENTS; ++agent) {
    auto& ad = fd_->agent_[agent];
    ad.planned_next_pos_ = assignment.next_positions[agent];
    ad.planned_next_best_path_ = assignment.next_paths[agent];
    ad.joint_expl_result_ = joint_result;
    ad.joint_assignment_pending_ = true;
    ad.replan_flag_ = true;
  }
  return true;
}

int ExplorationFSM::callActionPlanner(int agent_idx)
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;

  bool frontier_change_flag = updateFrontierAndObject();

  int expl_res, final_res;
  auto& ad = fd_->agent_[agent_idx];
  Eigen::Vector2d current_pos = Eigen::Vector2d(ad.start_pt_(0), ad.start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(ad.last_start_pos_(0), ad.last_start_pos_(1));
  double current_yaw = ad.start_yaw_;
  ad.last_start_pos_ = ad.start_pt_;
  const double planar_displacement = (current_pos - last_pos).norm();
  const int last_action = ad.newest_action_;
  const bool failed_forward = isFailedForwardAction(
      last_action, ACTION::MOVE_FORWARD, planar_displacement, stucking_distance);

  // Reach the object - check if close enough to target object
  if (ad.final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - ad.planned_next_pos_).norm() < reach_distance) {
    ROS_ERROR("Agent %d: Reach the object successfully!!!", agent_idx);
    final_res = FINAL_RESULT::REACH_OBJECT;
    return final_res;
  }

  // Escape-from-stuck logic
  if (!ad.escape_stucking_flag_ && failed_forward) {
    if (ad.final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - ad.planned_next_pos_).norm() < soft_reach_distance) {
      ROS_ERROR("Agent %d: Reach the object successfully!!!", agent_idx);
      final_res = FINAL_RESULT::REACH_OBJECT;
      return final_res;
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : ad.stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        ROS_ERROR("Agent %d: Still stuck at the same place", agent_idx);
        break;
      }
    }
    if (!past_stucking_flag) {
      ad.escape_stucking_flag_ = true;
      ad.escape_stucking_count_ = 0;
      ad.escape_stucking_pos_ = current_pos;
      ad.escape_stucking_yaw_ = current_yaw;
    }
    else {
      markForwardCollision(current_pos, current_yaw);
      ad.stucking_action_count_ = 0;
      ad.replan_flag_ = true;
      ad.dormant_frontier_flag_ = true;
    }
  }

  if (ad.escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    ROS_ERROR("Agent %d: Escaped from stuck state.", agent_idx);
    ad.escape_stucking_flag_ = false;
  }

  if (ad.escape_stucking_flag_) {
    ROS_ERROR("Agent %d: Escaping stuck...", agent_idx);
    if (ad.escape_stucking_count_ == 0)
      ad.newest_action_ = ACTION::TURN_RIGHT;
    else if (ad.escape_stucking_count_ == 1)
      ad.newest_action_ = ACTION::MOVE_FORWARD;
    else if (ad.escape_stucking_count_ == 2)
      ad.newest_action_ = ACTION::TURN_RIGHT;
    else if (ad.escape_stucking_count_ == 3)
      ad.newest_action_ = ACTION::MOVE_FORWARD;
    else if (ad.escape_stucking_count_ == 4)
      ad.newest_action_ = ACTION::TURN_LEFT;
    else if (ad.escape_stucking_count_ == 5)
      ad.newest_action_ = ACTION::TURN_LEFT;
    else if (ad.escape_stucking_count_ == 6)
      ad.newest_action_ = ACTION::TURN_LEFT;
    else if (ad.escape_stucking_count_ == 7)
      ad.newest_action_ = ACTION::MOVE_FORWARD;
    else if (ad.escape_stucking_count_ == 8)
      ad.newest_action_ = ACTION::TURN_LEFT;
    else if (ad.escape_stucking_count_ == 9)
      ad.newest_action_ = ACTION::MOVE_FORWARD;
    else {
      ad.escape_stucking_flag_ = false;
      markForwardCollision(ad.escape_stucking_pos_, ad.escape_stucking_yaw_);
      ad.stucking_action_count_ = 0;
      ad.replan_flag_ = true;
      ad.dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          ad.escape_stucking_pos_(0), ad.escape_stucking_pos_(1), ad.escape_stucking_yaw_);
      ad.stucking_points_.push_back(stucking_point);
    }

    if (ad.escape_stucking_flag_) {
      ad.escape_stucking_count_++;
      return ad.final_result_;
    }
  }

  // Replan path (stability heuristic) — use per-agent data
  vector<Vector2d> last_next_best_path = ad.planned_next_best_path_;
  Vector2d last_next_pos = ad.planned_next_pos_;
  if (ad.joint_assignment_pending_) {
    // Do not let the single-agent stability heuristic overwrite a fresh joint
    // assignment with the previous cycle's path.
    ad.replan_flag_ = true;
  }
  else if (ad.dormant_frontier_flag_) {
    ad.replan_flag_ = true;
    ad.dormant_frontier_flag_ = false;
  }
  else if (ad.final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag)
    ad.replan_flag_ = false;

  if (ad.joint_assignment_pending_) {
    // The synchronized solver provides only the first waypoint of each route;
    // normal action generation below still performs the rolling replan cycle.
    expl_res = ad.joint_expl_result_;
    ad.joint_assignment_pending_ = false;
  }
  else {
    expl_res = expl_manager_->planNextBestPoint(
        ad.start_pt_, ad.start_yaw_, agent_idx, ad.planned_next_pos_, ad.planned_next_best_path_);
  }
  ad.expl_result_ = expl_res;

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    ad.replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !ad.replan_flag_) {
    // Keep previous path — don't overwrite with new planning result
    ad.planned_next_best_path_ = last_next_best_path;
    ad.planned_next_pos_ = last_next_pos;
    ad.replan_flag_ = true;
  }

  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_.publish(expl_result_msg);
  publishExplorationResults();

  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  if (final_res == FINAL_RESULT::NO_FRONTIER || ad.planned_next_best_path_.empty()) {
    ROS_WARN("Agent %d: No (passable) frontier", agent_idx);
    return final_res;
  }

  Eigen::Vector2d end_pos = ad.planned_next_pos_;
  Eigen::Vector2d last_end_pos = ad.last_next_pos_;
  ad.last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  ROS_WARN("Agent %d: To the next point (%.2fm %.2fm), distance = %.2f m",
      agent_idx, end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      ROS_ERROR("Agent %d: Force set dormant frontier.", agent_idx);
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      ad.dormant_frontier_flag_ = true;
    }

    // Count consecutive times with same target position while stuck
    if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      ad.stucking_next_pos_count_++;
      ROS_ERROR_COND(ad.stucking_next_pos_count_ > 8, "Agent %d: stucking_next_pos_count_ = %d",
          agent_idx, ad.stucking_next_pos_count_);
    }
    else
      ad.stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (ad.stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      ROS_ERROR("Agent %d: Set dormant frontier.", agent_idx);
      ad.stucking_action_count_ = 0;
      ad.stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      ad.dormant_frontier_flag_ = true;
    }
  }

  // Track failed forward attempts only. Turns and camera actions are not
  // evidence that the robot is physically stuck.
  if (failed_forward) {
    ad.stucking_action_count_++;
    ROS_ERROR("Agent %d: Failed forward action count = %d", agent_idx,
        ad.stucking_action_count_);
  }
  else if (planar_displacement >= stucking_distance)
    ad.stucking_action_count_ = 0;

  // Plan specific action based on exploration result
  if (expl_res == EXPL_RESULT::SEARCH_EXTREME)
    ad.newest_action_ =
        planNextBestAction(current_pos, current_yaw, ad.planned_next_best_path_, false, agent_idx);
  else
    ad.newest_action_ =
        planNextBestAction(current_pos, current_yaw, ad.planned_next_best_path_, true, agent_idx);

  return final_res;
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety, int agent_idx)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->agent_[agent_idx].local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  return next_action;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  static int last_ftr2d_num = 0;
  static int last_dftr2d_num = 0;
  static int last_obj_num = 0;
  const size_t object_count = std::min(ed_ptr->objects_.size(), ed_ptr->object_labels_.size());
  if (ed_ptr->objects_.size() != ed_ptr->object_labels_.size()) {
    ROS_ERROR_THROTTLE(1.0, "object label count mismatch: objects=%zu labels=%zu",
        ed_ptr->objects_.size(), ed_ptr->object_labels_.size());
  }

  // Publish shared map markers (frontiers, objects, TSP tour) to EVERY agent's topic
  // so each agent's RViz panel receives them under its own topic namespace
  for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
    auto& agent_vis = visualization_[agent_idx];

    // Draw frontier
    for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
      agent_vis->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
          agent_vis->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
    }
    for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
      agent_vis->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    }

    // Draw dormant frontier
    for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
      agent_vis->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
          Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    }
    for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
      agent_vis->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    }

    // Draw object only where geometry and semantic labels agree.
    for (size_t i = 0; i < object_count; ++i) {
      int label = ed_ptr->object_labels_[i];
      agent_vis->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
          agent_vis->getColor(double(label) / 5.0, 1.0), "object", i, 4);
    }
    for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
      agent_vis->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
    }

    // Draw TSP tour (shared)
    agent_vis->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
        Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);
  }

  last_ftr2d_num = ed_ptr->frontiers_.size();
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();
  last_obj_num = object_count;

  // Draw per-agent trajectories and paths
  for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
    auto& agent_vis = visualization_[agent_idx];
    auto& ad = fd_->agent_[agent_idx];

    // Draw next best path for this agent (per-agent, decoupled)
    agent_vis->drawLines(vec2dTo3d(ad.planned_next_best_path_), fp_->vis_scale_,
        Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

    // Draw next local point for this agent
    vector<Vector2d> local_points;
    local_points.push_back(ad.local_pos_);
    agent_vis->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
        Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

    // Draw traveled path for this agent
    agent_vis->drawSpheres(vec2dTo3d(ad.traveled_path_), fp_->vis_scale_ * 1.5,
        Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
  }
}

void ExplorationFSM::clearVisMarker()
{
  for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
    auto& agent_vis = visualization_[agent_idx];
    for (int i = 0; i < 500; ++i) {
      agent_vis->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      agent_vis->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
      agent_vis->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
    }
    agent_vis->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
  }
}

bool ExplorationFSM::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  for (int i = 0; i < NUM_AGENTS; ++i) {
    if (!fd_->agent_[i].have_odom_)
      continue;
    const Eigen::Vector2d sensor_pos(
        fd_->agent_[i].odom_pos_(0), fd_->agent_[i].odom_pos_(1));
    change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->agent_[i].odom_yaw_);
  }
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

// Lightweight episode reset — only resets FSM state and agent data.
// Does NOT destroy/recreate ROS objects (timers, subscribers, publishers, maps).
// This is safe to call from any callback even with AsyncSpinner,
// unlike init(nh_) which destroys objects that other threads may be using.
void ExplorationFSM::resetEpisode()
{
  // Reset FSM state for all agents
  for (int i = 0; i < NUM_AGENTS; ++i)
    state_[i] = ROS_STATE::INIT;

  // Reset per-agent FSM data
  fd_.reset(new FSMData);

  // Reset exploration manager maps (SDF, frontier, object, value) without
  // destroying the ROS interface (MapROS subscribers/publishers stay alive).
  // Note: sdf_map_->resetMap() already resets object_map2d_ and value_map_.
  expl_manager_->sdf_map_->resetMap();
  expl_manager_->frontier_map2d_->reset();
  expl_manager_->ed_.reset(new ExplorationData);
  expl_manager_->resetEpisodeState();

  clearVisMarker();
  publishPlannerState();
  publishExplorationResults();
  ROS_WARN("Episode reset — FSM back to INIT, maps cleared.");
}

void ExplorationFSM::markForwardCollision(const Vector2d& origin, double yaw)
{
  for (int step = 1; step <= 2; ++step) {
    const double distance = FSMConstants::FORWARD_DISTANCE * step;
    Vector2d forward_pos = origin;
    forward_pos(0) += distance * cos(yaw);
    forward_pos(1) += distance * sin(yaw);
    expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
  }
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::Int32ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (msg->data == HABITAT_STATE::ACTION_FINISH) {
    // Trigger all agents that are waiting for action finish
    for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {
      if (state_[agent_idx] == ROS_STATE::WAIT_ACTION_FINISH) {
        transitState(agent_idx, ROS_STATE::PLAN_ACTION, "Habitat Finish Action");
      }
    }
  }
  if (msg->data == HABITAT_STATE::EPISODE_FINISH)
    resetEpisode();
  return;
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback(const ros::TimerEvent& e)
{
  bool all_wait = true;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    for (int i = 0; i < NUM_AGENTS; ++i) {
      if (state_[i] != ROS_STATE::WAIT_TRIGGER && state_[i] != ROS_STATE::FINISH &&
          state_[i] != ROS_STATE::FINISH_FAILURE) {
        all_wait = false;
        break;
      }
    }
  }
  if (!all_wait)
    return;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    updateFrontierAndObject();
    visualize();
  }
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Trigger all agents that are in WAIT_TRIGGER state
  bool any_triggered = false;
  for (int i = 0; i < NUM_AGENTS; ++i) {
    if (state_[i] == ROS_STATE::WAIT_TRIGGER) {
      fd_->agent_[i].trigger_ = true;
      transitState(i, ROS_STATE::PLAN_ACTION, "triggerCallback");
      any_triggered = true;
    }
  }
  if (any_triggered)
    cout << "Triggered all agents!" << endl;
}

void ExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg, int agent_idx)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto& ad = fd_->agent_[agent_idx];

  ad.odom_pos_(0) = msg->pose.pose.position.x;
  ad.odom_pos_(1) = msg->pose.pose.position.y;
  ad.odom_pos_(2) = msg->pose.pose.position.z;

  ad.odom_orient_.w() = msg->pose.pose.orientation.w;
  ad.odom_orient_.x() = msg->pose.pose.orientation.x;
  ad.odom_orient_.y() = msg->pose.pose.orientation.y;
  ad.odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = ad.odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  ad.odom_yaw_ = atan2(rot_x(1), rot_x(0));

  ad.have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(ad.odom_pos_(0), ad.odom_pos_(1));
  if (ad.traveled_path_.empty())
    ad.traveled_path_.push_back(odom_pos2d);
  else if ((ad.traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    ad.traveled_path_.push_back(odom_pos2d);

  publishRobotMarker(agent_idx);
}

void ExplorationFSM::publishRobotMarker(int agent_idx)
{
  auto& ad = fd_->agent_[agent_idx];
  const double robot_height = FSMConstants::ROBOT_HEIGHTS[agent_idx];
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  string agent_ns = "agent_" + std::to_string(agent_idx);

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = agent_ns + "_robot";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  robot_marker.pose.position.x = ad.odom_pos_(0);
  robot_marker.pose.position.y = ad.odom_pos_(1);
  robot_marker.pose.position.z = ad.odom_pos_(2) + robot_height / 2.0;

  robot_marker.pose.orientation.x = ad.odom_orient_.x();
  robot_marker.pose.orientation.y = ad.odom_orient_.y();
  robot_marker.pose.orientation.z = ad.odom_orient_.z();
  robot_marker.pose.orientation.w = ad.odom_orient_.w();

  robot_marker.scale.x = robot_radius * 2;
  robot_marker.scale.y = robot_radius * 2;
  robot_marker.scale.z = robot_height;

  const std::vector<Vector4d> body_colors = {
      Vector4d(50.0 / 255.0, 50.0 / 255.0, 255.0 / 255.0, 1.0),
      Vector4d(255.0 / 255.0, 50.0 / 255.0, 50.0 / 255.0, 1.0),
      Vector4d(50.0 / 255.0, 180.0 / 255.0, 80.0 / 255.0, 1.0)};
  const Vector4d& body_color = body_colors[agent_idx % body_colors.size()];
  robot_marker.color.r = body_color(0);
  robot_marker.color.g = body_color(1);
  robot_marker.color.b = body_color(2);
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = agent_ns + "_robot_dir";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  arrow_marker.pose.position.x = ad.odom_pos_(0);
  arrow_marker.pose.position.y = ad.odom_pos_(1);
  arrow_marker.pose.position.z = ad.odom_pos_(2) + robot_height;

  arrow_marker.pose.orientation.x = ad.odom_orient_.x();
  arrow_marker.pose.orientation.y = ad.odom_orient_.y();
  arrow_marker.pose.orientation.z = ad.odom_orient_.z();
  arrow_marker.pose.orientation.w = ad.odom_orient_.w();

  arrow_marker.scale.x = robot_radius + 0.13;
  arrow_marker.scale.y = 0.08;
  arrow_marker.scale.z = 0.08;

  const std::vector<Vector4d> arrow_colors = {
      Vector4d(10.0 / 255.0, 255.0 / 255.0, 10.0 / 255.0, 1.0),
      Vector4d(255.0 / 255.0, 165.0 / 255.0, 10.0 / 255.0, 1.0),
      Vector4d(40.0 / 255.0, 210.0 / 255.0, 255.0 / 255.0, 1.0)};
  const Vector4d& arrow_color = arrow_colors[agent_idx % arrow_colors.size()];
  arrow_marker.color.r = arrow_color(0);
  arrow_marker.color.g = arrow_color(1);
  arrow_marker.color.b = arrow_color(2);
  arrow_marker.color.a = 1.0;

  robot_marker_pub_[agent_idx].publish(robot_marker);
  robot_marker_pub_[agent_idx].publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

// Transition FSM state and log the change
// Caller must hold data_mutex_
void ExplorationFSM::transitState(int agent_idx, ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_[agent_idx]);
  state_[agent_idx] = new_state;
  cout << "[Agent " << agent_idx << " " + pos_call + "]: from " + stateName(pre_s) +
              " to " + stateName(int(new_state))
       << endl;
}
}  // namespace apexnav_planner
