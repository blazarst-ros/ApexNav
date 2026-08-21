#include <ros/ros.h>
#include <gcopter/trajectory.hpp>
#include <trajectory_manager/PolyTraj.h>
#include <Eigen/Dense>
#include <geometry_msgs/Twist.h>
#include <mavros_msgs/PositionTarget.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Empty.h>
#include <boost/function.hpp>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>
#include "controller/mpc.h"
#include <trajectory_manager/tracking_policy.h>
#include <trajectory_manager/trajectory_message_validation.h>

using namespace std;
using namespace Eigen;
// Execution accepts validated polynomial trajectories only; legacy rotation
// command bypasses have deliberately been removed.

class TrajectoryServer {
public:
  TrajectoryServer(ros::NodeHandle& nh)
  {
    nh_ = nh;
    receive_traj_ = false;
    have_odom_ = false;
    // Validate every MPC input before creating any ROS endpoint.  A constructor return
    // used to leave subscribers active while mpc_controller_ was null.
    std::vector<double> mpc_q;
    std::vector<double> mpc_r;
    std::vector<double> mpc_rd;
    int mpc_max_iter = 0;
    int mpc_delay_num = 0;
    double mpc_du_threshold = 0.0;
    const bool have_mpc_parameters = nh.getParam("mpc/predict_steps", mpc_N_) &&
                                     nh.getParam("mpc/dt", mpc_dt_) &&
                                     nh.getParam("mpc/max_iter", mpc_max_iter) &&
                                     nh.getParam("mpc/delay_num", mpc_delay_num) &&
                                     nh.getParam("mpc/du_threshold", mpc_du_threshold) &&
                                     nh.getParam("mpc/matrix_q", mpc_q) &&
                                     nh.getParam("mpc/matrix_r", mpc_r) &&
                                     nh.getParam("mpc/matrix_rd", mpc_rd) &&
                                     nh.getParam("max_correction_vel", max_correction_vel_) &&
                                     nh.getParam("max_correction_omega", max_correction_omega_);
    const auto finite_nonnegative = [](const std::vector<double>& values, size_t expected_size) {
      if (values.size() != expected_size)
        return false;
      for (const double value : values) {
        if (!std::isfinite(value) || value < 0.0)
          return false;
      }
      return true;
    };
    if (!have_mpc_parameters || mpc_N_ <= 0 || mpc_N_ >= 500 || !std::isfinite(mpc_dt_) ||
        mpc_dt_ <= 0.0 || mpc_max_iter <= 0 || mpc_delay_num < 0 || mpc_delay_num >= mpc_N_ ||
        !std::isfinite(mpc_du_threshold) || mpc_du_threshold < 0.0 ||
        !std::isfinite(max_correction_vel_) || max_correction_vel_ <= 0.0 ||
        !std::isfinite(max_correction_omega_) || max_correction_omega_ <= 0.0 ||
        !finite_nonnegative(mpc_q, 4) || !finite_nonnegative(mpc_r, 2) ||
        !finite_nonnegative(mpc_rd, 2)) {
      ROS_FATAL("[traj_server] Invalid or missing mandatory MPC parameters; shutting down");
      ros::shutdown();
      return;
    }

    nh.param("trajectory_max_stale_age_sec", trajectory_validation_policy_.max_stale_age_sec, 2.0);
    nh.param("trajectory_max_future_start_sec",
        trajectory_validation_policy_.max_future_start_sec, 1.0);
    if (!std::isfinite(trajectory_validation_policy_.max_stale_age_sec) ||
        trajectory_validation_policy_.max_stale_age_sec < 0.0 ||
        !std::isfinite(trajectory_validation_policy_.max_future_start_sec) ||
        trajectory_validation_policy_.max_future_start_sec < 0.0) {
      ROS_FATAL("[traj_server] Invalid trajectory message time validation policy; shutting down");
      ros::shutdown();
      return;
    }

    nh.param<std::string>("tracking_mode", tracking_mode_, "px4_native");
    nh.param("progress_full_rate_error", progress_full_rate_error_, 0.12);
    nh.param("progress_freeze_error", progress_freeze_error_, 0.25);
    nh.param("progress_minimum_rate", progress_minimum_rate_, 0.25);
    nh.param<std::string>("world_frame", world_frame_, "map");
    nh.param<std::string>("body_frame", body_frame_, "base_link");
    nh.param("odometry_max_stale_age_sec", odometry_validation_policy_.max_stale_age_sec, 0.20);
    nh.param("odometry_max_future_age_sec", odometry_validation_policy_.max_future_age_sec, 0.02);
    nh.param("odometry_quaternion_norm_tolerance",
        odometry_validation_policy_.quaternion_norm_tolerance, 0.05);
    odometry_validation_policy_.expected_world_frame = world_frame_;
    odometry_validation_policy_.expected_body_frame = body_frame_;
    const trajectory_manager::TrackingConfiguration tracking_configuration = {tracking_mode_,
        progress_full_rate_error_, progress_freeze_error_, progress_minimum_rate_};
    const auto tracking_configuration_validation =
        trajectory_manager::validateTrackingConfiguration(tracking_configuration);
    if (!tracking_configuration_validation.valid()) {
      ROS_FATAL("[traj_server] Invalid tracking configuration: %s; shutting down",
          trajectory_manager::trackingConfigurationValidationReasonString(
              tracking_configuration_validation.reason));
      ros::shutdown();
      return;
    }
    if (world_frame_.empty() || body_frame_.empty() ||
        !std::isfinite(odometry_validation_policy_.max_stale_age_sec) ||
        odometry_validation_policy_.max_stale_age_sec < 0.0 ||
        !std::isfinite(odometry_validation_policy_.max_future_age_sec) ||
        odometry_validation_policy_.max_future_age_sec < 0.0 ||
        !std::isfinite(odometry_validation_policy_.quaternion_norm_tolerance) ||
        odometry_validation_policy_.quaternion_norm_tolerance <= 0.0 ||
        odometry_validation_policy_.quaternion_norm_tolerance >= 1.0) {
      ROS_FATAL("[traj_server] Invalid odometry frame/time validation policy; shutting down");
      ros::shutdown();
      return;
    }

    subscribeTrajectory();
    odom_sub_ = nh_.subscribe("odometry", 10, &TrajectoryServer::odometryCallback, this);
    stop_sub_ = nh_.subscribe("/traj_server/stop", 10, &TrajectoryServer::stopCallback, this);
    robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);
    vel_cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 10);
    trajectory_reference_pub_ =
        nh_.advertise<mavros_msgs::PositionTarget>("trajectory_reference", 10);
    trajectory_progress_pub_ = nh_.advertise<std_msgs::Float64>("trajectory_progress", 10);
    traj_vis_pub_ = nh_.advertise<visualization_msgs::Marker>("/travel_traj", 10);
    current_desire_pub_ = nh_.advertise<geometry_msgs::Pose>("/current_desire", 10);
    vis_timer_ = nh_.createTimer(ros::Duration(0.20), &TrajectoryServer::visCallback, this);
    cmd_timer_ = nh_.createTimer(ros::Duration(0.02), &TrajectoryServer::cmdCallBack, this);
    std::cout << "[traj_server] TrajectoryServer initialized, waiting for messages..." << std::endl;

    mpc_controller_.reset(new MPC);
    mpc_controller_->init(nh_);
    xref_.resize(mpc_N_);
  }

  ros::Subscriber makeTrajectorySubscriber(std::uint64_t callback_transport_epoch)
  {
    // This NodeHandle uses the same callback queue as stopCallback. With ros::spin()
    // callbacks are serialized: a queued callback from the old subscriber retains the
    // old epoch and cannot commit state after stop/reset increments the epoch.
    boost::function<void(const trajectory_manager::PolyTrajConstPtr&)> callback =
        [this, callback_transport_epoch](const trajectory_manager::PolyTrajConstPtr& msg) {
          polyTrajCallback(msg, callback_transport_epoch);
        };
    ros::SubscribeOptions options = ros::SubscribeOptions::create<trajectory_manager::PolyTraj>(
        "trajectory", 10, callback, ros::VoidConstPtr(), nh_.getCallbackQueue());
    return nh_.subscribe(options);
  }

  void subscribeTrajectory()
  {
    traj_sub_ = makeTrajectorySubscriber(ingress_epoch_.transportEpoch());
  }

  template <typename EpochActivator>
  void rebuildTrajectorySubscriber(EpochActivator&& activate_epoch)
  {
    const std::uint64_t next_transport_epoch = ingress_epoch_.nextTransportEpoch();
    trajectory_manager::overlapSwapSubscriber(traj_sub_,
        [this, next_transport_epoch]() {
          return makeTrajectorySubscriber(next_transport_epoch);
        }, std::forward<EpochActivator>(activate_epoch));
  }

  void clearActiveTrajectory()
  {
    traj_.reset();
    start_time_ = ros::Time(0);
    traj_duration_ = 0.0;
    trajectory_progress_ = 0.0;
    last_command_time_ = ros::Time(0);
    receive_traj_ = false;
    ingress_epoch_.clearActiveTrajectory();
    if (mpc_controller_)
      mpc_controller_->reset();
  }

  void publishZeroVelocity()
  {
    geometry_msgs::Twist zero;
    vel_cmd_pub_.publish(zero);
  }

  void clearOdometryEpoch()
  {
    have_odom_ = false;
    last_odom_source_time_ = ros::Time(0);
  }

  bool odometryIsFresh(const ros::Time& now) const
  {
    if (!have_odom_ || last_odom_source_time_.isZero() || now.isZero())
      return false;
    const double age = (now - last_odom_source_time_).toSec();
    return std::isfinite(age) && age <= odometry_validation_policy_.max_stale_age_sec &&
           age >= -odometry_validation_policy_.max_future_age_sec;
  }

  void failClosedActiveTrajectory(const char* reason)
  {
    const bool had_active_command = receive_traj_ || ingress_epoch_.hasActiveTrajectory();
    clearActiveTrajectory();
    if (had_active_command) {
      publishZeroVelocity();
      ROS_ERROR("[traj_server] Trajectory failed closed: %s", reason);
    }
  }

  // Callback and 50 Hz command timer both use this path. A source-clock reset
  // discards the active trajectory before any further reference can be published.
  bool observeExecutionClock(const ros::Time& now)
  {
    const auto observation = ingress_epoch_.observeClock(now.toSec());
    if (observation == trajectory_manager::ClockObservation::kAccepted) {
      clock_invalid_latched_ = false;
      return true;
    }
    if (observation == trajectory_manager::ClockObservation::kInvalid && clock_invalid_latched_)
      return false;

    ROS_ERROR("[traj_server] Clearing trajectory: ROS clock is %s",
        observation == trajectory_manager::ClockObservation::kRegressed ? "regressed" : "zero/invalid");
    const bool had_active_command = receive_traj_ || ingress_epoch_.hasActiveTrajectory();
    clearActiveTrajectory();
    clearOdometryEpoch();
    traj_id_ = std::numeric_limits<int>::min();
    rebuildTrajectorySubscriber(
        [this, now]() { ingress_epoch_.beginNewClockEpoch(now.toSec()); });
    // A single zero is sent, then this server stays silent; supervisor HOLD owns
    // the continuing safe command stream.
    if (had_active_command)
      publishZeroVelocity();
    clock_invalid_latched_ = observation == trajectory_manager::ClockObservation::kInvalid;
    return false;
  }

  void polyTrajCallback(const trajectory_manager::PolyTrajConstPtr& msg,
      std::uint64_t callback_transport_epoch)
  {
    const ros::Time now = ros::Time::now();
    if (!ingress_epoch_.isCurrentTransportEpoch(callback_transport_epoch)) {
      ROS_WARN("[traj_server] Rejecting trajectory from stale transport epoch");
      return;
    }
    if (!observeExecutionClock(now))
      return;

    const trajectory_manager::TrajectoryMessageView<float> view = {msg->order,
        msg->start_time.toSec(), msg->coef_x, msg->coef_y, msg->coef_z, msg->duration};
    const auto validation = trajectory_manager::validateTrajectoryMessage(
        view, now.toSec(), trajectory_validation_policy_);
    if (!validation.valid()) {
      ROS_ERROR("[traj_server] Rejecting trajectory: %s",
          trajectory_manager::trajectoryMessageValidationReasonString(validation.reason));
      return;
    }

    if (ingress_epoch_.hasStopEpoch() && msg->start_time.toSec() <= ingress_epoch_.stopTimeSec()) {
      ROS_ERROR("[traj_server] Rejecting trajectory: start_time predates the last stop epoch");
      return;
    }
    if (msg->traj_id <= ingress_epoch_.lastAcceptedTrajectoryId()) {
      ROS_ERROR("[traj_server] Rejecting trajectory: id %d is not newer than last accepted id %d",
          msg->traj_id, ingress_epoch_.lastAcceptedTrajectoryId());
      return;
    }
    if (msg->start_time.toSec() < ingress_epoch_.lastAcceptedStartTimeSec()) {
      ROS_ERROR("[traj_server] Rejecting trajectory: source start_time %.6f regressed from %.6f",
          msg->start_time.toSec(), ingress_epoch_.lastAcceptedStartTimeSec());
      return;
    }
    if (!ingress_epoch_.canAcceptTrajectory(
            callback_transport_epoch, msg->traj_id, msg->start_time.toSec())) {
      ROS_ERROR("[traj_server] Rejecting trajectory: transport/stop epoch barrier");
      return;
    }

    int piece_nums = msg->duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<Eigen::Matrix<double, 3, 8>> cMats(piece_nums);

    for (int i = 0; i < piece_nums; ++i) {
      int i8 = i * 8;
      cMats[i].row(0) << msg->coef_x[i8 + 0], msg->coef_x[i8 + 1], msg->coef_x[i8 + 2],
          msg->coef_x[i8 + 3], msg->coef_x[i8 + 4], msg->coef_x[i8 + 5], msg->coef_x[i8 + 6],
          msg->coef_x[i8 + 7];
      cMats[i].row(1) << msg->coef_y[i8 + 0], msg->coef_y[i8 + 1], msg->coef_y[i8 + 2],
          msg->coef_y[i8 + 3], msg->coef_y[i8 + 4], msg->coef_y[i8 + 5], msg->coef_y[i8 + 6],
          msg->coef_y[i8 + 7];
      cMats[i].row(2) << msg->coef_z[i8 + 0], msg->coef_z[i8 + 1], msg->coef_z[i8 + 2],
          msg->coef_z[i8 + 3], msg->coef_z[i8 + 4], msg->coef_z[i8 + 5], msg->coef_z[i8 + 6],
          msg->coef_z[i8 + 7];
      dura[i] = msg->duration[i];
    }

    std::unique_ptr<Trajectory<7, 3>> next_traj;
    try {
      next_traj.reset(new Trajectory<7, 3>(dura, cMats));
    }
    catch (const std::exception& error) {
      ROS_ERROR("[traj_server] Rejecting trajectory: construction failed: %s", error.what());
      return;
    }

    const double next_duration = next_traj->getTotalDuration();
    if (!std::isfinite(next_duration) || next_duration <= 0.0) {
      ROS_ERROR("[traj_server] Rejecting trajectory: constructed duration is invalid");
      return;
    }

    traj_ = std::move(next_traj);
    start_time_ = msg->start_time;
    traj_duration_ = next_duration;
    traj_id_ = msg->traj_id;
    ingress_epoch_.noteAcceptedTrajectory(msg->traj_id, msg->start_time.toSec());
    trajectory_progress_ = 0.0;
    last_command_time_ = ros::Time(0);
    last_reference_yaw_ = odometryIsFresh(now) ? odom_yaw_ : 0.0;
    mpc_controller_->reset();
    receive_traj_ = true;

    std::cout << "[traj_server] Received trajectory ID " << traj_id_
              << ", total duration: " << traj_duration_ << ", start_time: " << start_time_.toSec()
              << std::endl;
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg)
  {
    const ros::Time now = ros::Time::now();
    if (!observeExecutionClock(now))
      return;

    const std::array<double, 3> position = {{msg->pose.pose.position.x,
        msg->pose.pose.position.y, msg->pose.pose.position.z}};
    const std::array<double, 4> orientation_xyzw = {{msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y, msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w}};
    const std::array<double, 6> twist = {{msg->twist.twist.linear.x,
        msg->twist.twist.linear.y, msg->twist.twist.linear.z,
        msg->twist.twist.angular.x, msg->twist.twist.angular.y,
        msg->twist.twist.angular.z}};
    const trajectory_manager::OdometrySampleView view = {msg->header.stamp.toSec(),
        msg->header.frame_id, msg->child_frame_id, position.data(), position.size(),
        orientation_xyzw.data(), orientation_xyzw.size(), twist.data(), twist.size(),
        msg->pose.covariance.data(), msg->pose.covariance.size(),
        msg->twist.covariance.data(), msg->twist.covariance.size()};
    const auto validation = trajectory_manager::validateOdometrySample(view, now.toSec(),
        last_odom_source_time_.toSec(), odometry_validation_policy_);
    if (!validation.valid()) {
      ROS_WARN_THROTTLE(1.0, "[traj_server] Rejecting odometry: %s",
          trajectory_manager::odometrySampleValidationReasonString(validation.reason));
      return;
    }

    const Eigen::Vector3d next_position(position[0], position[1], position[2]);
    Eigen::Quaterniond next_orientation(orientation_xyzw[3], orientation_xyzw[0],
        orientation_xyzw[1], orientation_xyzw[2]);
    next_orientation.normalize();
    const Eigen::Vector3d next_linear_velocity(twist[0], twist[1], twist[2]);
    const Eigen::Vector3d rot_x = next_orientation.toRotationMatrix().block<3, 1>(0, 0);
    const double next_yaw = atan2(rot_x(1), rot_x(0));
    if (!next_position.allFinite() || !next_orientation.coeffs().allFinite() ||
        !next_linear_velocity.allFinite() || !std::isfinite(next_yaw)) {
      ROS_WARN_THROTTLE(1.0, "[traj_server] Rejecting odometry: derived state is non-finite");
      return;
    }

    // Commit only after the entire source sample and every derived value have
    // passed validation, so a corrupt packet cannot partially poison state.
    odom_pos_ = next_position;
    odom_orient_ = next_orientation;
    odom_linear_vel_ = next_linear_velocity;
    odom_yaw_ = next_yaw;
    last_odom_source_time_ = msg->header.stamp;
    have_odom_ = true;
    // publishRobotMarker();
    traj_real_.push_back(Eigen::Vector3d(odom_pos_(0), odom_pos_(1), 0.15));
    if (traj_real_.size() > 50000)
      traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 10000);
  }

  void stopCallback(const std_msgs::EmptyConstPtr& msg)
  {
    const ros::Time now = ros::Time::now();
    if (!observeExecutionClock(now))
      return;
    rebuildTrajectorySubscriber(
        [this, now]() { ingress_epoch_.beginStopEpoch(now.toSec()); });
    clearActiveTrajectory();
    // Publish one zero command only. The supervisor's 30 Hz HOLD stream owns the
    // persistent safe command after this ingress server goes silent.
    publishZeroVelocity();
    ROS_WARN_THROTTLE(1.0, "[traj_server] Trajectory cleared; one zero command published");
  }

  void visCallback(const ros::TimerEvent& e)
  {
    displayTrajWithColor(
        traj_real_, 0.10, Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), 0);
  }

  void cmdCallBack(const ros::TimerEvent& event)
  {
    const ros::Time current_time = ros::Time::now();
    if (!observeExecutionClock(current_time))
      return;

    if (!receive_traj_ || !ingress_epoch_.hasActiveTrajectory()) {
      return;
    }

    double elapsed_time = (current_time - start_time_).toSec();

    if (elapsed_time < 0)
      return;  // Wait for start time to pass

    if (!odometryIsFresh(current_time)) {
      failClosedActiveTrajectory("odometry is missing or stale");
      return;
    }

    if (tracking_mode_ == "px4_native") {
      publishNativeReference(current_time);
      return;
    }

    if (elapsed_time > traj_duration_) {
      // Trajectory finished, stop publishing
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
      vel_cmd_pub_.publish(twist_msg);  // Publish zero velocity
      clearActiveTrajectory();           // Reset flag so that no more commands are published
      return;
    }

    Eigen::Vector3d pos = traj_->getPos(elapsed_time);
    Eigen::Vector3d vel = traj_->getVel(elapsed_time);
    if (!pos.allFinite() || !vel.allFinite()) {
      failClosedActiveTrajectory("legacy trajectory evaluation is non-finite");
      return;
    }

    Eigen::Vector3d ref;
    ref << pos(0), pos(1), atan2(vel(1), vel(0));
    for (int i = 0; i < mpc_N_; ++i) {
      double temp_t = elapsed_time + i * mpc_dt_;
      if (temp_t <= traj_duration_) {
        pos = traj_->getPos(temp_t);
        vel = traj_->getVel(temp_t);
        if (!pos.allFinite() || !vel.allFinite()) {
          failClosedActiveTrajectory("legacy prediction reference is non-finite");
          return;
        }
        ref << pos(0), pos(1), atan2(vel(1), vel(0));
      }
      if (!ref.allFinite()) {
        failClosedActiveTrajectory("legacy prediction yaw is non-finite");
        return;
      }
      xref_[i] = ref;
    }
    mpc_controller_->setOdom(
        Eigen::Vector4d(odom_pos_(0), odom_pos_(1), odom_yaw_, odom_linear_vel_.head(2).norm()));
    const Eigen::Vector2d cmd = mpc_controller_->calCmd(xref_);
    if (!cmd.allFinite()) {
      failClosedActiveTrajectory("legacy MPC output is non-finite");
      return;
    }
    geometry_msgs::Twist twist_msg;
    twist_msg.linear.x = cmd(0);
    twist_msg.angular.z = cmd(1);
    vel_cmd_pub_.publish(twist_msg);
    publishDesiredPose(traj_->getPos(elapsed_time), traj_->getVel(elapsed_time));
  }

  void publishDesiredPose(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity)
  {
    if (velocity.head(2).norm() > 0.02)
      last_reference_yaw_ = atan2(velocity(1), velocity(0));
    geometry_msgs::Pose desire_pose;
    desire_pose.position.x = position(0);
    desire_pose.position.y = position(1);
    desire_pose.position.z = position(2);
    const Eigen::Quaterniond q(
        Eigen::AngleAxisd(last_reference_yaw_, Eigen::Vector3d::UnitZ()));
    desire_pose.orientation.x = q.x();
    desire_pose.orientation.y = q.y();
    desire_pose.orientation.z = q.z();
    desire_pose.orientation.w = q.w();
    current_desire_pub_.publish(desire_pose);
  }

  void publishNativeReference(const ros::Time& now)
  {
    double dt = 0.0;
    if (!last_command_time_.isZero())
      dt = std::max(0.0, std::min(0.10, (now - last_command_time_).toSec()));
    last_command_time_ = now;

    const Eigen::Vector3d tracking_position = traj_->getPos(trajectory_progress_);
    const double tracking_error =
        (tracking_position.head(2) - odom_pos_.head(2)).norm();
    if (!tracking_position.allFinite() || !std::isfinite(tracking_error)) {
      failClosedActiveTrajectory("native tracking state is non-finite");
      return;
    }
    const double rate = std::max(0.0, std::min(1.0, trajectory_manager::progressRate(tracking_error,
        progress_full_rate_error_, progress_freeze_error_, progress_minimum_rate_)));
    trajectory_progress_ = std::min(traj_duration_, trajectory_progress_ + rate * dt);

    const Eigen::Vector3d position = traj_->getPos(trajectory_progress_);
    Eigen::Vector3d velocity = traj_->getVel(trajectory_progress_) * rate;
    const Eigen::Vector3d acceleration = traj_->getAcc(trajectory_progress_) * rate * rate;
    if (!std::isfinite(trajectory_progress_) || !std::isfinite(rate) || !position.allFinite() ||
        !velocity.allFinite() || !acceleration.allFinite()) {
      failClosedActiveTrajectory("native trajectory reference is non-finite");
      return;
    }
    const double planar_speed_sq = velocity.head(2).squaredNorm();
    double yaw_rate = 0.0;
    if (planar_speed_sq > 4e-4) {
      last_reference_yaw_ = atan2(velocity(1), velocity(0));
      yaw_rate = (velocity(0) * acceleration(1) - velocity(1) * acceleration(0)) /
                 (planar_speed_sq + 1e-3);
    }
    if (!std::isfinite(last_reference_yaw_) || !std::isfinite(yaw_rate)) {
      failClosedActiveTrajectory("native yaw reference is non-finite");
      return;
    }

    mavros_msgs::PositionTarget reference;
    reference.header.stamp = now;
    reference.header.frame_id = world_frame_;
    reference.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    reference.type_mask = mavros_msgs::PositionTarget::IGNORE_VZ |
                          mavros_msgs::PositionTarget::IGNORE_AFX |
                          mavros_msgs::PositionTarget::IGNORE_AFY |
                          mavros_msgs::PositionTarget::IGNORE_AFZ;
    reference.position.x = position(0);
    reference.position.y = position(1);
    reference.position.z = position(2);
    reference.velocity.x = velocity(0);
    reference.velocity.y = velocity(1);
    reference.yaw = last_reference_yaw_;
    reference.yaw_rate = yaw_rate;
    trajectory_reference_pub_.publish(reference);

    std_msgs::Float64 progress;
    progress.data = trajectory_progress_;
    trajectory_progress_pub_.publish(progress);
    publishDesiredPose(position, velocity);

    ROS_WARN_THROTTLE(1.0,
        "[traj_server] Native tracking progress %.2f/%.2fs, error %.3fm, rate %.2f",
        trajectory_progress_, traj_duration_, tracking_error, rate);

    if (trajectory_progress_ >= traj_duration_ - 1e-6)
      clearActiveTrajectory();
  }

  void publishRobotMarker()
  {
    const double robot_height = 0.15;
    const double robot_radius = 0.18;

    visualization_msgs::Marker marker;
    marker.header.frame_id = world_frame_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "robot_position";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CYLINDER;  // Set to CYLINDER
    marker.action = visualization_msgs::Marker::ADD;

    // Set cylinder position
    marker.pose.position.x = odom_pos_(0);
    marker.pose.position.y = odom_pos_(1);
    marker.pose.position.z = odom_pos_(2) + robot_height / 2.0;

    // Set cylinder orientation (quaternion)
    marker.pose.orientation.x = odom_orient_.x();
    marker.pose.orientation.y = odom_orient_.y();
    marker.pose.orientation.z = odom_orient_.z();
    marker.pose.orientation.w = odom_orient_.w();

    // Set cylinder size
    marker.scale.x = robot_radius * 2;  // diameter
    marker.scale.y = robot_radius * 2;  // diameter
    marker.scale.z = robot_height;      // height

    marker.color.r = 50.0 / 255.0;
    marker.color.g = 50.0 / 255.0;
    marker.color.b = 255.0 / 255.0;
    marker.color.a = 1.0;  // opaque

    // Create and publish arrow (direction)
    visualization_msgs::Marker arrow_marker;
    arrow_marker.header.frame_id = world_frame_;
    arrow_marker.header.stamp = ros::Time::now();
    arrow_marker.ns = "robot_direction";
    arrow_marker.id = 1;
    arrow_marker.type = visualization_msgs::Marker::ARROW;  // Set to ARROW
    arrow_marker.action = visualization_msgs::Marker::ADD;

    // Set arrow position (start)
    arrow_marker.pose.position.x = odom_pos_(0);
    arrow_marker.pose.position.y = odom_pos_(1);
    arrow_marker.pose.position.z = odom_pos_(2) + robot_height;

    // Set arrow orientation (from quaternion)
    arrow_marker.pose.orientation.x = odom_orient_.x();
    arrow_marker.pose.orientation.y = odom_orient_.y();
    arrow_marker.pose.orientation.z = odom_orient_.z();
    arrow_marker.pose.orientation.w = odom_orient_.w();

    // Set arrow size
    arrow_marker.scale.x = robot_radius + 0.13;  // arrow length
    arrow_marker.scale.y = 0.08;                 // arrow width
    arrow_marker.scale.z = 0.08;                 // arrow thickness

    arrow_marker.color.r = 10.0 / 255.0;
    arrow_marker.color.g = 255.0 / 255.0;
    arrow_marker.color.b = 10.0 / 255.0;
    arrow_marker.color.a = 1.0;  // opaque

    robot_marker_pub_.publish(marker);
    robot_marker_pub_.publish(arrow_marker);
  }

  void displayTrajWithColor(
      vector<Eigen::Vector3d> path, double resolution, Eigen::Vector4d color, int id)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = world_frame_;
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::SPHERE_LIST;
    mk.action = visualization_msgs::Marker::DELETE;
    mk.id = id;
    traj_vis_pub_.publish(mk);

    mk.action = visualization_msgs::Marker::ADD;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = color(3);
    mk.scale.x = resolution;
    mk.scale.y = resolution;
    mk.scale.z = resolution;
    geometry_msgs::Point pt;
    for (int i = 0; i < int(path.size()); i++) {
      pt.x = path[i](0);
      pt.y = path[i](1);
      pt.z = path[i](2);
      mk.points.push_back(pt);
    }
    traj_vis_pub_.publish(mk);
    ros::Duration(0.0001).sleep();
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber traj_sub_, odom_sub_, stop_sub_;
  ros::Publisher vel_cmd_pub_, trajectory_reference_pub_, trajectory_progress_pub_;
  ros::Publisher robot_marker_pub_, traj_vis_pub_, current_desire_pub_;
  ros::Timer cmd_timer_, vis_timer_;

  // Trajectory Data
  std::unique_ptr<Trajectory<7, 3>> traj_;
  ros::Time start_time_;
  double traj_duration_;
  int traj_id_ = std::numeric_limits<int>::min();
  bool receive_traj_;
  bool clock_invalid_latched_ = false;
  trajectory_manager::TrajectoryIngressEpoch ingress_epoch_;
  trajectory_manager::TrajectoryMessageValidationPolicy trajectory_validation_policy_;
  trajectory_manager::OdometrySampleValidationPolicy odometry_validation_policy_;
  std::string tracking_mode_;
  double trajectory_progress_ = 0.0;
  double progress_full_rate_error_ = 0.12;
  double progress_freeze_error_ = 0.25;
  double progress_minimum_rate_ = 0.25;
  double last_reference_yaw_ = 0.0;
  ros::Time last_command_time_;

  bool use_mpc_ = true;
  MPC::Ptr mpc_controller_;
  std::vector<Eigen::Vector3d> xref_;
  int mpc_N_;
  double mpc_dt_;

  // Data
  Vector3d odom_pos_, odom_linear_vel_;
  Quaterniond odom_orient_;
  double odom_yaw_;
  bool have_odom_;
  ros::Time last_odom_source_time_;
  double replan_time_ = 0.5;
  vector<Eigen::Vector3d> traj_real_;
  double max_correction_vel_, max_correction_omega_;
  std::string world_frame_;
  std::string body_frame_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_server_node");
  ros::NodeHandle nh("~");
  TrajectoryServer traj_server(nh);
  ros::spin();
  return 0;
}
