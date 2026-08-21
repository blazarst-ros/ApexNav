#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm_traj.h>
#include <exploration_manager/local_target_selector.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>

#include <stdexcept>
#include <utility>

namespace apexnav_planner {

void ExplorationFSMReal::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstantsReal::VIS_SCALE_FACTOR;

  state_ = RealFSM::State::INIT;

  // Load real-world specific parameters
  nh.param("fsm/replan_time", fp_->replan_time_, 0.2);
  nh.param("fsm/replan_traj_end_threshold", fp_->replan_traj_end_threshold_, 1.0);
  nh.param("fsm/replan_frontier_change_delay", fp_->replan_frontier_change_delay_, 0.5);
  nh.param("fsm/replan_timeout", fp_->replan_timeout_, 2.0);
  nh.param("fsm/replan_wall_timeout", fp_->replan_wall_timeout_, 12.0);
  nh.param("fsm/replan_min_execution_time", fp_->replan_min_execution_time_, 2.5);
  nh.param("fsm/replan_min_progress", fp_->replan_min_progress_, 0.25);
  nh.param("fsm/local_target_distance", fp_->local_target_distance_, 1.0);
  nh.param("fsm/local_target_min_progress", fp_->local_target_min_progress_, 0.10);
  nh.param("fsm/plan_failure_retry_sec", fp_->plan_failure_retry_sec_, 0.25);
  nh.param("fsm/tracking_reanchor_threshold", fp_->tracking_reanchor_threshold_, 0.25);
  nh.param("fsm/tracking_error_threshold", fp_->tracking_error_threshold_, 0.40);
  nh.param("fsm/tracking_error_hold_sec", fp_->tracking_error_hold_sec_, 0.60);
  nh.param("fsm/trajectory_speed_limit", fp_->trajectory_speed_limit_, 0.22);
  nh.param("fsm/trajectory_yaw_rate_limit", fp_->trajectory_yaw_rate_limit_, 0.35);
  nh.param("fsm/replan_settle_speed", fp_->replan_settle_speed_, 0.05);
  nh.param("fsm/replan_settle_sec", fp_->replan_settle_sec_, 0.50);
  nh.param("fsm/trajectory_start_tolerance", fp_->trajectory_start_tolerance_, 0.10);
  nh.param("fsm/target_lock_sec", fp_->target_lock_sec_, 10.0);
  nh.param("fsm/target_reached_distance", fp_->target_reached_distance_, 0.35);
  nh.param("fsm/target_failure_limit", fp_->target_failure_limit_, 3);
  nh.param("fsm/target_failure_min_hysteresis", target_failure_min_hysteresis_, 2.0);
  nh.param<std::string>("map_ros/frame_id", world_frame_, "map");
  nh.param<std::string>("fsm/base_frame", base_frame_, "base_link");
  nh.param("fsm/trigger_max_age", trigger_max_age_, 1.00);
  nh.param("fsm/odometry_max_age", odometry_max_age_, 0.20);
  nh.param("fsm/input_max_future", input_max_future_, 0.05);
  nh.param("fsm/odometry_quaternion_norm_tolerance",
      odometry_quaternion_norm_tolerance_, 0.05);
  nh.param("fsm/first_trajectory_progress_limit", first_trajectory_progress_limit_, 0.25);
  if (world_frame_.empty() || base_frame_.empty() ||
      !std::isfinite(fp_->plan_failure_retry_sec_) || fp_->plan_failure_retry_sec_ <= 0.0 ||
      fp_->target_failure_limit_ <= 0 || !std::isfinite(target_failure_min_hysteresis_) ||
      target_failure_min_hysteresis_ < 0.0 ||
      !std::isfinite(trigger_max_age_) ||
      trigger_max_age_ <= 0.0 || !std::isfinite(odometry_max_age_) ||
      odometry_max_age_ <= 0.0 || !std::isfinite(input_max_future_) ||
      input_max_future_ < 0.0 || !std::isfinite(odometry_quaternion_norm_tolerance_) ||
      odometry_quaternion_norm_tolerance_ < 0.0 ||
      !std::isfinite(first_trajectory_progress_limit_) ||
      first_trajectory_progress_limit_ < 0.0) {
    ROS_FATAL("[Real] Invalid planner ingress/frame configuration");
    throw std::runtime_error("invalid planner ingress/frame configuration");
  }
  ROS_INFO("[Real] Target failure hysteresis: at least %d failures and %.2fs wall time",
      fp_->target_failure_limit_, target_failure_min_hysteresis_);
  last_trigger_source_stamp_ = ros::Time(0);
  last_odometry_source_stamp_ = ros::Time(0);
  next_plan_attempt_time_ = ros::Time(0);
  tracking_error_since_ = ros::Time(0);
  trajectory_execution_start_ = ros::Time(0);
  trajectory_start_odom_.setZero();
  trajectory_progress_ = 0.0;
  last_trajectory_progress_ = ros::Time(0);
  trajectory_progress_started_ = false;
  require_replan_settle_ = false;
  have_locked_target_ = false;
  locked_target_.setZero();
  locked_target_result_ = EXPL_RESULT::EXPLORATION;
  locked_target_failures_ = 0;
  locked_target_since_ = ros::Time(0);
  locked_target_first_failure_since_ = ros::Time(0);

  /* ROS Timer */
  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstantsReal::EXEC_TIMER_DURATION), &ExplorationFSMReal::FSMCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstantsReal::FRONTIER_TIMER_DURATION),
      &ExplorationFSMReal::frontierCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &ExplorationFSMReal::safetyCallback, this);

  /* ROS Subscriber */
  trigger_sub_ =
      nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSMReal::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 10, &ExplorationFSMReal::odometryCallback, this);
  confidence_threshold_sub_ = nh.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSMReal::confidenceThresholdCallback, this);
  cancel_sub_ =
      nh.subscribe("/apexnav/planner/cancel", 10, &ExplorationFSMReal::cancelCallback, this);
  trajectory_progress_sub_ = nh.subscribe("/apexnav/planner/trajectory_progress", 20,
      &ExplorationFSMReal::trajectoryProgressCallback, this);
  navigation_enabled_sub_ = nh.subscribe("/apexnav/mission/navigation_enabled", 2,
      &ExplorationFSMReal::navigationEnabledCallback, this);

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);

  // Real-world trajectory publishers
  poly_traj_pub_ = nh.advertise<trajectory_manager::PolyTraj>("/planning/trajectory", 10);
  stop_pub_ = nh.advertise<std_msgs::Empty>("/traj_server/stop", 10);

  ROS_INFO("[ExplorationFSMReal] Initialization complete.");
}

// Main FSM callback for real-world exploration
void ExplorationFSMReal::FSMCallback(const ros::TimerEvent& e)
{
  exec_timer_.stop();

  // Publish current state
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = static_cast<int>(state_);
  ros_state_pub_.publish(ros_state_msg);

  switch (state_) {
    case RealFSM::State::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        ROS_WARN_THROTTLE(1.0, "[Real] No odom || No target confidence threshold.");
        exec_timer_.start();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(RealFSM::State::WAIT_TRIGGER, "FSM");
      break;
    }

    case RealFSM::State::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "[Real] Waiting for trigger...");
      break;
    }

    case RealFSM::State::FINISH: {
      fd_->static_state_ = true;
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        emergencyStop();
        clearVisMarker();
      }
      ROS_WARN_THROTTLE(1.0, "[Real] Finish exploration!");
      break;
    }

    case RealFSM::State::PLAN_TRAJ: {
      if (!next_plan_attempt_time_.isZero() && ros::Time::now() < next_plan_attempt_time_)
        break;

      if (require_replan_settle_) {
        const double speed = fd_->odom_vel_.head(2).norm();
        if (!replan_settle_gate_.update(ros::Time::now().toSec(), speed,
                fp_->replan_settle_speed_, fp_->replan_settle_sec_)) {
          ROS_WARN_THROTTLE(0.5,
              "[Real] Brake-and-hold before replanning: vxy %.3f/%.3fm/s",
              speed, fp_->replan_settle_speed_);
          break;
        }
        require_replan_settle_ = false;
        ROS_INFO("[Real] Vehicle settled for %.2fs; replanning from measured odometry",
            fp_->replan_settle_sec_);
      }

      // The PX4 supervisor deliberately position-holds during PLAN_TRAJ. Plan
      // from measured odometry with a stopped boundary condition, matching
      // that execution contract. Predicting along the previous trajectory
      // while the supervisor holds causes every replan start to drift ahead
      // of the aircraft.
      if (!fd_->static_state_) {
        LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
        double old_t = std::max(0.0, (ros::Time::now() - info->start_time).toSec());
        old_t = std::min(old_t, info->duration);
        const double old_tracking_error =
            (info->traj.getPos(old_t).head(2) - fd_->odom_pos_.head(2)).norm();
        ROS_INFO("[Real] Stop-and-replan from odometry; old trajectory error %.3fm",
            old_tracking_error);
      }
      fd_->start_pt_ = fd_->odom_pos_;
      fd_->start_vel_.setZero();
      fd_->start_yaw_(0) = fd_->odom_yaw_;
      fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
      fd_->static_state_ = true;

      TrajPlannerResult res = callTrajectoryPlanner();

      if (res == TrajPlannerResult::RETRYABLE_FAILED) {
        next_plan_attempt_time_ = ros::Time::now() + ros::Duration(fp_->plan_failure_retry_sec_);
        ROS_WARN_THROTTLE(1.0, "[Real] Plan trajectory failed; retrying after %.2fs",
            fp_->plan_failure_retry_sec_);
        fd_->static_state_ = true;
      }
      else if (res == TrajPlannerResult::SUCCESS) {
        next_plan_attempt_time_ = ros::Time(0);
        transitState(RealFSM::State::EXEC_TRAJ, "FSM");
      }
      else {
        // Both terminal outcomes have already stopped the stream in the
        // planner.  Repeat the stop at the state boundary before FINISH so
        // terminal handling remains fail-closed if this function changes.
        emergencyStop();
        ROS_WARN("[Real] Navigation terminal result: %s",
            res == TrajPlannerResult::MISSION_SUCCEEDED ? "succeeded" : "failed");
        transitState(RealFSM::State::FINISH, "FSM");
      }

      visualize();
      break;
    }

    case RealFSM::State::EXEC_TRAJ: {
      // Publish trajectory and transition to execution monitoring
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        trajectory_manager::PolyTraj poly_msg;
        polyTraj2ROSMsg(fd_->newest_traj_, poly_msg);
        poly_traj_pub_.publish(poly_msg);
        trajectory_execution_start_ = ros::Time::now();
        trajectory_start_odom_ = fd_->odom_pos_.head(2);
        trajectory_progress_ = 0.0;
        last_trajectory_progress_ = ros::Time(0);
        trajectory_progress_started_ = false;
        fd_->static_state_ = false;
        transitState(RealFSM::State::REPLAN, "FSM");
      }
      break;
    }

    case RealFSM::State::REPLAN: {
      // Monitor trajectory execution and decide when to replan
      LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
      double t_cur = std::max(0.0, (ros::Time::now() - info->start_time).toSec());
      if (!last_trajectory_progress_.isZero() &&
          (ros::Time::now() - last_trajectory_progress_).toSec() <= 0.30)
        t_cur = std::max(0.0, std::min(trajectory_progress_, info->duration));
      double time_to_end = info->duration - t_cur;
      const double execution_age = trajectory_execution_start_.isZero()
          ? std::max(0.0, t_cur)
          : (ros::Time::now() - trajectory_execution_start_).toSec();
      const double execution_progress =
          (fd_->odom_pos_.head(2) - trajectory_start_odom_).norm();

      // Replan if trajectory is almost finished
      if (time_to_end < fp_->replan_traj_end_threshold_) {
        emergencyStop();
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: traj fully executed");
        exec_timer_.start();
        return;
      }

      // Continuous depth mapping changes some frontier on nearly every frame.
      // Do not discard a valid active trajectory because an unrelated
      // frontier changed elsewhere. Replan early only when the active local
      // target itself has become unsafe, and only after a minimum committed
      // execution interval/progress. The trajectory safety timer remains an
      // unconditional immediate obstacle stop.
      const bool execution_committed =
          execution_age >= std::max(
              fp_->replan_min_execution_time_, fp_->replan_frontier_change_delay_) &&
          execution_progress >= fp_->replan_min_progress_;
      const Eigen::Vector2d active_local_target = expl_manager_->ed_->next_local_pos_;
      const bool active_target_invalid =
          expl_manager_->sdf_map_->getInflateOccupancy(active_local_target);
      if (execution_committed && active_target_invalid) {
        emergencyStop();
        have_locked_target_ = false;
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: active local target invalid after %.2fs / %.2fm",
            execution_age, execution_progress);
        exec_timer_.start();
        return;
      }

      // Replan if trajectory timeout
      if (t_cur > fp_->replan_timeout_) {
        emergencyStop();
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: periodic refresh after %.2fs / %.2fm", execution_age,
            execution_progress);
        exec_timer_.start();
        return;
      }
      if (execution_age > fp_->replan_wall_timeout_) {
        emergencyStop();
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: wall-time guard after %.2fs with trajectory progress %.2fs",
            execution_age, t_cur);
        exec_timer_.start();
        return;
      }
      break;
    }
  }

  exec_timer_.start();
}

void ExplorationFSMReal::recordLockedTargetFailure()
{
  if (locked_target_failures_ == 0 || locked_target_first_failure_since_.isZero())
    locked_target_first_failure_since_ = ros::Time::now();
  if (locked_target_failures_ < std::numeric_limits<int>::max())
    ++locked_target_failures_;
}

void ExplorationFSMReal::resetLockedTargetFailures()
{
  locked_target_failures_ = 0;
  locked_target_first_failure_since_ = ros::Time(0);
}

TrajPlannerResult ExplorationFSMReal::callTrajectoryPlanner()
{
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);
  updateFrontierAndObject();

  int expl_res = EXPL_RESULT::EXPLORATION;
  bool reused_locked_target = false;
  const ros::Time now = ros::Time::now();
  const double target_distance =
      have_locked_target_ ? (fd_->start_pt_.head(2) - locked_target_).norm() : 0.0;
  const bool target_unsafe = have_locked_target_ &&
      expl_manager_->sdf_map_->getInflateOccupancy(locked_target_);
  const double lock_age = (have_locked_target_ && !locked_target_since_.isZero())
      ? (now - locked_target_since_).toSec()
      : 0.0;
  const double failure_age = (locked_target_failures_ > 0 &&
      !locked_target_first_failure_since_.isZero())
      ? std::max(0.0, (now - locked_target_first_failure_since_).toSec())
      : 0.0;

  if (shouldReuseLockedTarget(have_locked_target_, target_distance, target_unsafe,
          lock_age, fp_->target_lock_sec_, locked_target_failures_,
          fp_->target_failure_limit_, fp_->target_reached_distance_, failure_age,
          target_failure_min_hysteresis_)) {
    if (expl_manager_->planPathToLockedTarget(fd_->start_pt_, locked_target_)) {
      expl_res = locked_target_result_;
      reused_locked_target = true;
      ROS_INFO("[Real] Reusing locked target (%.2f, %.2f), age %.2fs",
          locked_target_.x(), locked_target_.y(), lock_age);
    }
    else {
      recordLockedTargetFailure();
      ROS_WARN("[Real] Locked target path failed (%d/%d, %.2f/%.2fs); "
               "preserving target hysteresis",
          locked_target_failures_, fp_->target_failure_limit_,
          (ros::Time::now() - locked_target_first_failure_since_).toSec(),
          target_failure_min_hysteresis_);
      return TrajPlannerResult::RETRYABLE_FAILED;
    }
  }
  else if (have_locked_target_) {
    ROS_INFO("[Real] Releasing locked target: distance %.2fm, unsafe=%s, age %.2fs, "
             "failures=%d for %.2fs",
        target_distance, target_unsafe ? "true" : "false", lock_age,
        locked_target_failures_, failure_age);
    have_locked_target_ = false;
    resetLockedTargetFailures();
  }

  if (!reused_locked_target) {
    expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));
    if (expl_res != EXPL_RESULT::NO_COVERABLE_FRONTIER &&
        expl_res != EXPL_RESULT::NO_PASSABLE_FRONTIER &&
        !expl_manager_->ed_->next_best_path_.empty()) {
      have_locked_target_ = true;
      locked_target_ = expl_manager_->ed_->next_pos_;
      locked_target_result_ = expl_res;
      resetLockedTargetFailures();
      locked_target_since_ = now;
      ROS_INFO("[Real] Locked new navigation target (%.2f, %.2f) for %.2fs",
          locked_target_.x(), locked_target_.y(), fp_->target_lock_sec_);
    }
  }

  // Determine final result based on exploration result
  if (expl_res == EXPL_RESULT::EXPLORATION)
    fd_->final_result_ = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    fd_->final_result_ = FINAL_RESULT::NO_FRONTIER;
  else
    fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;

  // Publish exploration result
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = fd_->final_result_;
  expl_result_pub_.publish(expl_result_msg);

  if (terminalNavigationResult(fd_->final_result_) ==
      TerminalNavigationResult::MISSION_FAILED) {
    ROS_WARN("[Real] No (passable) frontier");
    have_locked_target_ = false;
    resetLockedTargetFailures();
    emergencyStop();
    return TrajPlannerResult::MISSION_FAILED;
  }

  // Preserve the object approach pose selected by the global planner. The
  // local selector below is allowed to replace goal_pos with a safe prefix;
  // that prefix must never be interpreted as having reached the object.
  const Eigen::Vector2d final_object_approach_goal = expl_manager_->ed_->next_pos_;
  const bool approaching_object = fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT;
  if (approaching_object && !final_object_approach_goal.allFinite()) {
    ROS_ERROR("[Real] Rejecting object approach with non-finite global target");
    if (have_locked_target_)
      recordLockedTargetFailure();
    return TrajPlannerResult::RETRYABLE_FAILED;
  }
  if (approaching_object && hasReachedFinalObjectApproachGoal(
          fd_->odom_pos_(0), fd_->odom_pos_(1), final_object_approach_goal.x(),
          final_object_approach_goal.y(), 0.25)) {
    ROS_ERROR("[Real] Reached final object approach goal successfully!");
    fd_->final_result_ = FINAL_RESULT::REACH_OBJECT;
    have_locked_target_ = false;
    resetLockedTargetFailures();
    expl_result_msg.data = fd_->final_result_;
    expl_result_pub_.publish(expl_result_msg);
    emergencyStop();
    return TrajPlannerResult::MISSION_SUCCEEDED;
  }

  // Select local target from global path.
  Eigen::Vector2d goal_pos = final_object_approach_goal;
  double goal_yaw = 0.0;
  auto path = expl_manager_->ed_->next_best_path_;
  if (!selectLocalTarget(
          fd_->start_pt_.head(2), path, fp_->local_target_distance_, goal_pos, goal_yaw)) {
    ROS_WARN_THROTTLE(1.0,
        "[Real] Global path has no contiguous footprint-safe local target within %.2fm; "
        "rejecting it instead of jumping across the obstacle",
        fp_->local_target_distance_);
    if (have_locked_target_)
      recordLockedTargetFailure();
    return TrajPlannerResult::RETRYABLE_FAILED;
  }

  // Prepare state for trajectory planning
  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control(0.0, 0.0, 0.0);
  double start_vel = Eigen::Vector2d(fd_->start_vel_(0), fd_->start_vel_(1)).norm();
  current_state << fd_->start_pt_(0), fd_->start_pt_(1), fd_->start_yaw_(0), 0.0, start_vel;
  goal_state << goal_pos(0), goal_pos(1), goal_yaw, 0.0, 0.0;

  // Plan trajectory using GCopter
  bool traj_res = expl_manager_->planTrajectory(current_state, goal_state, current_control);
  if (traj_res) {
    auto info = &expl_manager_->gcopter_->local_trajectory_;
    if (!validateTrajectoryStructure(*info, "before trajectory evaluation", nullptr)) {
      if (have_locked_target_)
        recordLockedTargetFailure();
      return TrajPlannerResult::RETRYABLE_FAILED;
    }
    const double start_error =
        (info->traj.getPos(0.0).head(2) - fd_->odom_pos_.head(2)).norm();
    if (!trajectoryStartIsContinuous(start_error, fp_->trajectory_start_tolerance_)) {
      ROS_ERROR("[Real] Rejecting discontinuous trajectory start %.3fm > %.3fm",
          start_error, fp_->trajectory_start_tolerance_);
      if (have_locked_target_)
        recordLockedTargetFailure();
      return TrajPlannerResult::RETRYABLE_FAILED;
    }
    if (!enforceTrajectoryLimits(*info)) {
      if (have_locked_target_)
        recordLockedTargetFailure();
      return TrajPlannerResult::RETRYABLE_FAILED;
    }
    if (!validateTrajectoryStructure(*info, "after trajectory limiting", nullptr)) {
      if (have_locked_target_)
        recordLockedTargetFailure();
      return TrajPlannerResult::RETRYABLE_FAILED;
    }
    if (!validateFinalTrajectory(*info)) {
      if (have_locked_target_)
        recordLockedTargetFailure();
      return TrajPlannerResult::RETRYABLE_FAILED;
    }
    resetLockedTargetFailures();
    info->start_time = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;
    fd_->newest_traj_ = expl_manager_->gcopter_->local_trajectory_;
    return TrajPlannerResult::SUCCESS;
  }

  if (have_locked_target_)
    recordLockedTargetFailure();
  return TrajPlannerResult::RETRYABLE_FAILED;
}

bool ExplorationFSMReal::validateTrajectoryStructure(
    const LocalTrajectory& local_traj, const char* stage, double* total_duration) const
{
  const int piece_num = local_traj.traj.getPieceNum();
  if (piece_num <= 0) {
    ROS_ERROR("[Real] Rejecting trajectory %s: no polynomial pieces", stage);
    return false;
  }

  std::vector<TrajectoryPieceLayout> layout;
  layout.reserve(piece_num);
  for (int piece_index = 0; piece_index < piece_num; ++piece_index) {
    const auto& piece = local_traj.traj[piece_index];
    const auto& coefficients = piece.getCoeffMat();
    TrajectoryPieceLayout metadata;
    metadata.duration = piece.getDuration();
    metadata.coefficients.assign(coefficients.data(), coefficients.data() + coefficients.size());
    layout.push_back(std::move(metadata));
  }
  if (!hasValidTrajectoryPieceLayout(layout)) {
    ROS_ERROR("[Real] Rejecting trajectory %s: invalid piece duration or coefficient", stage);
    return false;
  }

  if (total_duration != nullptr) {
    *total_duration = 0.0;
    for (const auto& piece : layout)
      *total_duration += piece.duration;
    if (!std::isfinite(*total_duration) || *total_duration <= 0.0) {
      ROS_ERROR("[Real] Rejecting trajectory %s: invalid total duration", stage);
      return false;
    }
  }
  return true;
}

bool ExplorationFSMReal::enforceTrajectoryLimits(LocalTrajectory& local_traj)
{
  const double speed_limit = std::max(0.01, fp_->trajectory_speed_limit_);
  const double yaw_rate_limit = std::max(0.01, fp_->trajectory_yaw_rate_limit_);
  double duration = 0.0;
  if (!validateTrajectoryStructure(local_traj, "before limit enforcement", &duration))
    return false;
  double max_speed = 0.0;
  double max_yaw_rate = 0.0;

  // Derive proof-carrying Bernstein bounds from every polynomial piece. A
  // fixed 0.02 s scan can alias a finite degree-six spike whose position and
  // velocity both vanish at every scan instant. The derivative convex hull
  // and subdivided rational yaw-rate enclosure cover the complete interval.
  for (int piece_index = 0; piece_index < local_traj.traj.getPieceNum(); ++piece_index) {
    const auto& piece = local_traj.traj[piece_index];
    const double piece_duration = piece.getDuration();
    const auto& coefficients = piece.getCoeffMat();
    std::vector<std::vector<double>> planar_power_coefficients(
        2, std::vector<double>(coefficients.cols(), 0.0));
    for (int dimension = 0; dimension < 2; ++dimension) {
      for (int power = 0; power < coefficients.cols(); ++power) {
        planar_power_coefficients[dimension][power] =
            coefficients(dimension, coefficients.cols() - 1 - power);
      }
    }

    double piece_speed_bound = 0.0;
    double piece_yaw_rate_bound = 0.0;
    if (!conservativeBernsteinDerivativeNormBound(
            planar_power_coefficients, piece_duration, &piece_speed_bound) ||
        !conservativeBernsteinPlanarYawRateBound(
            planar_power_coefficients, piece_duration, 1e-3, 8,
            &piece_yaw_rate_bound)) {
      ROS_ERROR("[Real] Rejecting trajectory during limit enforcement: "
                "non-finite polynomial derivative bound");
      return false;
    }
    max_speed = std::max(max_speed, piece_speed_bound);
    max_yaw_rate = std::max(max_yaw_rate, piece_yaw_rate_bound);
    if (!std::isfinite(max_speed) || !std::isfinite(max_yaw_rate)) {
      ROS_ERROR("[Real] Rejecting trajectory during limit enforcement: non-finite limit bound");
      return false;
    }
  }

  const double scale = std::max(
      1.0, std::max(max_speed / speed_limit, max_yaw_rate / yaw_rate_limit));
  if (!std::isfinite(scale) || scale <= 0.0) {
    ROS_ERROR("[Real] Rejecting trajectory during limit enforcement: invalid scale");
    return false;
  }
  if (scale <= 1.0 + 1e-12) {
    local_traj.duration = duration;
    return validateTrajectoryStructure(local_traj, "after limit enforcement", nullptr);
  }

  std::vector<double> scaled_durations;
  std::vector<Piece<7, 3>::CoefficientMat> scaled_coefficients;
  const Eigen::VectorXd durations = local_traj.traj.getDurations();
  scaled_durations.reserve(local_traj.traj.getPieceNum());
  scaled_coefficients.reserve(local_traj.traj.getPieceNum());
  for (int i = 0; i < local_traj.traj.getPieceNum(); ++i) {
    const double scaled_duration = durations(i) * scale;
    if (!std::isfinite(scaled_duration) || scaled_duration <= 0.0) {
      ROS_ERROR("[Real] Rejecting trajectory during limit enforcement: invalid scaled duration");
      return false;
    }
    scaled_durations.push_back(scaled_duration);
    Piece<7, 3>::CoefficientMat coefficients = local_traj.traj[i].getCoeffMat();
    for (int column = 0; column < 8; ++column) {
      const int power = 7 - column;
      const double scale_power = std::pow(scale, power);
      if (!std::isfinite(scale_power) || scale_power <= 0.0) {
        ROS_ERROR("[Real] Rejecting trajectory during limit enforcement: invalid scale power");
        return false;
      }
      coefficients.col(column) /= scale_power;
    }
    scaled_coefficients.push_back(coefficients);
  }
  local_traj.traj = Trajectory<7, 3>(scaled_durations, scaled_coefficients);
  double scaled_total_duration = 0.0;
  if (!validateTrajectoryStructure(local_traj, "after limit enforcement", &scaled_total_duration))
    return false;
  local_traj.duration = scaled_total_duration;
  ROS_WARN("[Real] Time-scaled trajectory %.2fx (bounded raw vmax %.3fm/s, "
           "yaw-rate %.3frad/s; "
           "limits %.3f/%.3f)",
      scale, max_speed, max_yaw_rate, speed_limit, yaw_rate_limit);
  return true;
}

bool ExplorationFSMReal::validateFinalTrajectory(const LocalTrajectory& local_traj) const
{
  double total_duration = 0.0;
  if (!validateTrajectoryStructure(local_traj, "before final safety validation", &total_duration) ||
      !std::isfinite(fd_->odom_yaw_)) {
    ROS_ERROR("[Real] Rejecting final trajectory: invalid structure or odometry yaw");
    return false;
  }

  const int piece_num = local_traj.traj.getPieceNum();
  std::vector<double> piece_durations;
  std::vector<double> piece_derivative_bounds;
  piece_durations.reserve(piece_num);
  piece_derivative_bounds.reserve(piece_num);
  for (int piece_index = 0; piece_index < piece_num; ++piece_index) {
    const auto& piece = local_traj.traj[piece_index];
    const double piece_duration = piece.getDuration();
    piece_durations.push_back(piece_duration);
    const auto& coefficients = piece.getCoeffMat();
    std::vector<std::vector<double>> power_coefficients(
        coefficients.rows(), std::vector<double>(coefficients.cols(), 0.0));
    for (int dimension = 0; dimension < coefficients.rows(); ++dimension) {
      for (int power = 0; power < coefficients.cols(); ++power) {
        power_coefficients[dimension][power] =
            coefficients(dimension, coefficients.cols() - 1 - power);
      }
    }
    double derivative_bound = 0.0;
    if (!conservativeBernsteinDerivativeNormBound(
            power_coefficients, piece_duration, &derivative_bound)) {
      ROS_ERROR("[Real] Rejecting final trajectory: invalid derivative bound for piece %d",
          piece_index);
      return false;
    }
    piece_derivative_bounds.push_back(derivative_bound);
  }
  const auto evaluator = [&local_traj](std::size_t piece_index, double local_time) {
    const Eigen::VectorXd position =
        local_traj.traj[static_cast<int>(piece_index)].getPos(local_time);
    return TrajectorySamplePoint{position(0), position(1), position(2)};
  };
  const AdaptiveFinalTrajectorySampling sampling = makeAdaptiveFinalTrajectorySamples(
      piece_durations, piece_derivative_bounds, evaluator,
      0.05, 0.05, 1e-6, 32, 1000000);
  if (!sampling.valid || sampling.samples.empty() ||
      std::abs(sampling.samples.back().global_time - total_duration) > 1e-9) {
    ROS_ERROR("[Real] Rejecting final trajectory: adaptive safety sampling failed");
    return false;
  }

  double previous_yaw = fd_->odom_yaw_;
  for (const AdaptiveTrajectorySample& sample : sampling.samples) {
    const Eigen::Vector3d pos(
        sample.position.x, sample.position.y, sample.position.z);
    const Eigen::Vector3d vel =
        local_traj.traj[static_cast<int>(sample.piece_index)].getVel(sample.local_time);
    if (!pos.allFinite() || !vel.allFinite()) {
      ROS_ERROR("[Real] Rejecting final trajectory: non-finite pos/vel at time %.3f",
          sample.global_time);
      return false;
    }

    const double yaw = yawForTrajectorySample(vel.x(), vel.y(), previous_yaw);
    if (!std::isfinite(yaw)) {
      ROS_ERROR("[Real] Rejecting final trajectory: non-finite yaw at time %.3f",
          sample.global_time);
      return false;
    }
    previous_yaw = yaw;

    const Eigen::Vector2d pos_2d = pos.head(2);
    const int inflated_occupancy = expl_manager_->sdf_map_->getInflateOccupancy(pos_2d);
    const bool footprint_collision = expl_manager_->kinoastar_->isCollisionPosYaw(pos_2d, yaw);
    // getInflateOccupancy returns -1 outside the map; every non-zero value is
    // unsafe, including unknown/out-of-map and inflated occupied cells.
    if (inflated_occupancy != 0 || footprint_collision) {
      ROS_ERROR("[Real] Rejecting final trajectory: %s at (%.2f, %.2f), yaw %.2f, time %.3f",
          footprint_collision ? "footprint collision" : "inflation/unknown collision",
          pos_2d.x(), pos_2d.y(), yaw, sample.global_time);
      return false;
    }
  }
  return true;
}

void ExplorationFSMReal::polyTraj2ROSMsg(
    const LocalTrajectory& local_traj, trajectory_manager::PolyTraj& poly_msg)
{
  auto data = &local_traj;
  Eigen::VectorXd durs = data->traj.getDurations();
  int piece_num = data->traj.getPieceNum();

  poly_msg.drone_id = 0;
  poly_msg.traj_id = data->traj_id;
  poly_msg.start_time = data->start_time;
  poly_msg.order = 7;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);

  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);

    auto cMat = data->traj.operator[](i).getCoeffMat();
    int i8 = i * 8;
    for (int j = 0; j < 8; j++) {
      poly_msg.coef_x[i8 + j] = cMat(0, j);
      poly_msg.coef_y[i8 + j] = cMat(1, j);
      poly_msg.coef_z[i8 + j] = cMat(2, j);
    }
  }
}

bool ExplorationFSMReal::selectLocalTarget(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path, const double& local_distance,
    Eigen::Vector2d& target_pos, double& target_yaw)
{
  if (path.empty()) {
    target_pos = current_pos;
    target_yaw = 0.0;
    expl_manager_->ed_->next_local_pos_ = target_pos;
    ROS_WARN("Local target path is empty; holding current collision-free pose.");
    return false;
  }

  const auto collision = [this](const Eigen::Vector2d& position, double yaw) {
    return expl_manager_->kinoastar_->isCollisionPosYaw(position, yaw);
  };
  const LocalTargetSelection selected =
      selectFootprintSafeLocalTarget(
          current_pos, path, local_distance, fp_->local_target_min_progress_, collision);
  if (!selected.valid) {
    target_pos = current_pos;
    target_yaw = 0.0;
    expl_manager_->ed_->next_local_pos_ = target_pos;
    return false;
  }
  target_pos = selected.position;
  target_yaw = selected.yaw;

  // Gradient-based safety adjustment
  const Eigen::Vector2d footprint_safe_pos = target_pos;
  const double footprint_safe_yaw = target_yaw;
  double step_size = 0.05;
  double tolerance = 1e-3;
  int max_iterations = 30;

  for (int i = 0; i < max_iterations; ++i) {
    Eigen::Vector2d prev_pos = target_pos;

    // Get gradient from SDF map
    Eigen::Vector2d grad;
    double dist = expl_manager_->sdf_map_->getDistWithGrad(target_pos, grad);

    if (dist > 0.26)
      break;

    // Move along gradient to safer position
    if (grad.norm() > 1e-6) {
      target_pos += step_size * grad.normalized();
    }

    // Check convergence
    if ((target_pos - prev_pos).norm() < tolerance) {
      break;
    }
  }

  if (expl_manager_->kinoastar_->isCollisionPosYaw(target_pos, target_yaw)) {
    target_pos = footprint_safe_pos;
    target_yaw = footprint_safe_yaw;
  }

  // Store selected local target
  expl_manager_->ed_->next_local_pos_ = target_pos;
  return true;
}

void ExplorationFSMReal::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  auto vec2dTo3d = [](const std::vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    std::vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Eigen::Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontiers
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontiers
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw objects
  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Eigen::Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  std::vector<Eigen::Vector2d> local_points;
  local_points.push_back(ed_ptr->next_local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Eigen::Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Eigen::Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);
}

void ExplorationFSMReal::clearVisMarker()
{
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  visualization_->drawLines({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSMReal::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d sensor_pos = Eigen::Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->odom_yaw_);
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

void ExplorationFSMReal::frontierCallback(const ros::TimerEvent& e)
{
  // Update frontiers and visualize in idle states
  if (state_ != RealFSM::State::WAIT_TRIGGER && state_ != RealFSM::State::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

void ExplorationFSMReal::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if (!shouldAcceptPlannerTrigger(state_ == RealFSM::State::WAIT_TRIGGER,
          navigation_enabled_)) {
    ROS_WARN_THROTTLE(1.0,
        "[Real] Ignoring planner trigger while navigation is disabled or FSM is not ready");
    return;
  }
  const ros::Time now = ros::Time::now();
  if (msg->header.frame_id != world_frame_ ||
      !sourceStampIsAcceptable(now.toSec(), msg->header.stamp.toSec(),
          last_trigger_source_stamp_.toSec(), trigger_max_age_, input_max_future_)) {
    ROS_WARN_THROTTLE(1.0,
        "[Real] Rejecting planner trigger with stale/replayed stamp or frame '%s' (expected '%s')",
        msg->header.frame_id.c_str(), world_frame_.c_str());
    return;
  }
  last_trigger_source_stamp_ = msg->header.stamp;

  fd_->trigger_ = true;
  trajectory_execution_start_ = ros::Time(0);
  trajectory_start_odom_ = fd_->odom_pos_.head(2);
  trajectory_progress_ = 0.0;
  last_trajectory_progress_ = ros::Time(0);
  trajectory_progress_started_ = false;
  tracking_error_since_ = ros::Time(0);
  have_locked_target_ = false;
  resetLockedTargetFailures();
  require_replan_settle_ = true;
  replan_settle_gate_.reset();
  ROS_INFO("[Real] Exploration triggered!");
  transitState(RealFSM::State::PLAN_TRAJ, "triggerCallback");
}

void ExplorationFSMReal::navigationEnabledCallback(const std_msgs::BoolConstPtr& msg)
{
  navigation_enabled_ = msg->data;
}

void ExplorationFSMReal::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
  const ros::Time now = ros::Time::now();
  const auto& position = msg->pose.pose.position;
  const auto& orientation = msg->pose.pose.orientation;
  const auto& linear = msg->twist.twist.linear;
  const auto& angular = msg->twist.twist.angular;
  const bool finite_state = std::isfinite(position.x) && std::isfinite(position.y) &&
      std::isfinite(position.z) && std::isfinite(linear.x) && std::isfinite(linear.y) &&
      std::isfinite(linear.z) && std::isfinite(angular.x) && std::isfinite(angular.y) &&
      std::isfinite(angular.z);
  if (msg->header.frame_id != world_frame_ || msg->child_frame_id != base_frame_ ||
      !sourceStampIsAcceptable(now.toSec(), msg->header.stamp.toSec(),
          last_odometry_source_stamp_.toSec(), odometry_max_age_, input_max_future_) ||
      !finite_state || !quaternionIsFiniteAndNormalized(orientation.x, orientation.y,
          orientation.z, orientation.w, odometry_quaternion_norm_tolerance_)) {
    ROS_WARN_THROTTLE(1.0,
        "[Real] Rejecting invalid odometry source (frame '%s'/'%s', expected '%s'/'%s')",
        msg->header.frame_id.c_str(), msg->child_frame_id.c_str(),
        world_frame_.c_str(), base_frame_.c_str());
    return;
  }
  last_odometry_source_stamp_ = msg->header.stamp;

  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;
  fd_->odom_orient_.normalize();

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  // Extract linear velocity
  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  // Extract angular velocity
  fd_->odom_omega_(0) = msg->twist.twist.angular.x;
  fd_->odom_omega_(1) = msg->twist.twist.angular.y;
  fd_->odom_omega_(2) = msg->twist.twist.angular.z;

  fd_->have_odom_ = true;

  // Publish robot marker for visualization
  publishRobotMarker();
}

void ExplorationFSMReal::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (fd_->have_confidence_)
    return;
  if (!confidenceThresholdIsValid(msg->data)) {
    ROS_ERROR_THROTTLE(1.0,
        "[Real] Rejecting non-finite/out-of-range confidence threshold");
    return;
  }
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
  ROS_INFO("[Real] Confidence threshold set to: %.2f", msg->data);
}

void ExplorationFSMReal::cancelCallback(const std_msgs::EmptyConstPtr& msg)
{
  if (state_ == RealFSM::State::FINISH)
    return;

  ROS_WARN("[Real] Mission cancelled; stopping trajectory generation.");
  emergencyStop();
  trajectory_execution_start_ = ros::Time(0);
  tracking_error_since_ = ros::Time(0);
  have_locked_target_ = false;
  resetLockedTargetFailures();
  fd_->have_finished_ = true;
  clearVisMarker();
  transitState(RealFSM::State::FINISH, "cancelCallback");
}

void ExplorationFSMReal::emergencyStop()
{
  fd_->static_state_ = true;
  trajectory_progress_started_ = false;
  last_trajectory_progress_ = ros::Time(0);
  require_replan_settle_ = true;
  replan_settle_gate_.reset();
  stop_pub_.publish(std_msgs::Empty());
}

void ExplorationFSMReal::trajectoryProgressCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (state_ != RealFSM::State::REPLAN || fd_->static_state_)
    return;
  const double trajectory_duration = expl_manager_->gcopter_->local_trajectory_.duration;
  if (!trajectoryProgressIsValid(msg->data, trajectory_progress_, trajectory_duration,
          trajectory_progress_started_, first_trajectory_progress_limit_)) {
    ROS_WARN_THROTTLE(1.0,
        "[Real] Rejecting invalid/replayed trajectory progress %.3f (previous %.3f, duration %.3f)",
        msg->data, trajectory_progress_, trajectory_duration);
    return;
  }
  trajectory_progress_ = std::min(msg->data, trajectory_duration);
  trajectory_progress_started_ = true;
  last_trajectory_progress_ = ros::Time::now();
}

void ExplorationFSMReal::safetyCallback(const ros::TimerEvent& e)
{
  if (state_ != RealFSM::State::REPLAN) {
    tracking_error_since_ = ros::Time(0);
    return;
  }

  // Check if robot deviates from planned trajectory
  double t_cur = std::max(0.0,
      (ros::Time::now() - expl_manager_->gcopter_->local_trajectory_.start_time).toSec());
  if (!last_trajectory_progress_.isZero() &&
      (ros::Time::now() - last_trajectory_progress_).toSec() <= 0.30)
    t_cur = trajectory_progress_;
  t_cur = min(t_cur, expl_manager_->gcopter_->local_trajectory_.duration);
  Eigen::Vector3d cur_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_cur);

  const double tracking_error = (cur_pos.head(2) - fd_->odom_pos_.head(2)).norm();
  if (tracking_error > fp_->tracking_error_threshold_) {
    ROS_ERROR("[Real] Hard tracking error %.3fm exceeds %.3fm; stopping and reanchoring",
        tracking_error, fp_->tracking_error_threshold_);
    emergencyStop();
    transitState(RealFSM::State::PLAN_TRAJ, "Hard Tracking Error");
    tracking_error_since_ = ros::Time(0);
    return;
  }
  if (tracking_error > fp_->tracking_reanchor_threshold_) {
    if (tracking_error_since_.isZero())
      tracking_error_since_ = ros::Time::now();
    const double held = (ros::Time::now() - tracking_error_since_).toSec();
    ROS_WARN_THROTTLE(0.5,
        "[Real] Tracking error %.3fm exceeds reanchor threshold %.3fm "
        "(persistent %.2f/%.2fs)", tracking_error, fp_->tracking_reanchor_threshold_, held,
        fp_->tracking_error_hold_sec_);
    if (held >= fp_->tracking_error_hold_sec_) {
      ROS_WARN("[Real] Persistent tracking error %.3fm; stopping and replanning from odometry",
          tracking_error);
      emergencyStop();
      transitState(RealFSM::State::PLAN_TRAJ, "Tracking Reanchor");
      tracking_error_since_ = ros::Time(0);
      return;
    }
  }
  else {
    tracking_error_since_ = ros::Time(0);
  }

  // Combine a conservative centre clearance check with an exact oriented
  // footprint check. Runtime safety therefore remains independent of the
  // global planner's tunable inflation radius.
  double time_horizon = 2.5;  // Check trajectory for next 2.5 seconds
  double sample_dt = 0.1;     // Sample every 0.1 seconds

  double check_yaw = fd_->odom_yaw_;
  for (double t_check = t_cur;
      t_check <= min(t_cur + time_horizon, expl_manager_->gcopter_->local_trajectory_.duration);
      t_check += sample_dt) {
    Eigen::Vector3d check_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_check);
    const Eigen::Vector2d check_vel =
        expl_manager_->gcopter_->local_trajectory_.traj.getVel(t_check).head(2);
    Eigen::Vector2d check_pos_2d = check_pos.head(2);
    check_yaw = yawForTrajectorySample(check_vel.x(), check_vel.y(), check_yaw);

    const bool centre_clearance_violation =
        expl_manager_->sdf_map_->getInflateOccupancy(check_pos_2d);
    const bool footprint_collision =
        expl_manager_->kinoastar_->isCollisionPosYaw(check_pos_2d, check_yaw);
    if (centre_clearance_violation || footprint_collision) {
      ROS_ERROR("[Real] Safety Stop!!! %s at (%.2f, %.2f), yaw %.2f, time %.2f",
          footprint_collision ? "footprint collision" : "clearance violation",
          check_pos_2d(0), check_pos_2d(1), check_yaw, t_check);
      have_locked_target_ = false;
      emergencyStop();
      transitState(RealFSM::State::PLAN_TRAJ, "Trajectory Safety Stop");
      break;
    }
  }
}

void ExplorationFSMReal::publishRobotMarker()
{
  const double robot_height = FSMConstantsReal::ROBOT_HEIGHT;
  const double robot_radius = FSMConstantsReal::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = world_frame_;
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  robot_marker.scale.x = robot_radius * 2;
  robot_marker.scale.y = robot_radius * 2;
  robot_marker.scale.z = robot_height;

  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = world_frame_;
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  arrow_marker.scale.x = robot_radius + 0.13;
  arrow_marker.scale.y = 0.08;
  arrow_marker.scale.z = 0.08;

  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSMReal::transitState(RealFSM::State new_state, std::string pos_call)
{
  std::string state_str[] = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "EXEC_TRAJ", "REPLAN",
    "FINISH" };
  ROS_INFO("[Real FSM]: %s -> from %s to %s", pos_call.c_str(),
      state_str[static_cast<int>(state_)].c_str(), state_str[static_cast<int>(new_state)].c_str());
  state_ = new_state;
}

}  // namespace apexnav_planner
