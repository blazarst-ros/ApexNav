#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace trajectory_manager {

// Replace a subscription without ever removing the final live handle first.
// roscpp coalesces callbacks for an identical topic/type onto the existing
// transport subscription, so registering the replacement before shutting down
// the old callback avoids a disconnect/reconnect delivery gap.
template <typename Subscriber, typename Factory, typename Activator>
void overlapSwapSubscriber(
    Subscriber& active, Factory&& create_replacement, Activator&& activate_replacement)
{
  Subscriber replacement = std::forward<Factory>(create_replacement)();
  // The replacement callback is registered while the old epoch is still
  // current.  With the node's serialized callback queue it cannot execute
  // until this function returns, so activating its epoch here closes the
  // receive window that would otherwise exist between an epoch increment and
  // registering the replacement.
  std::forward<Activator>(activate_replacement)();
  active.shutdown();
  active = std::move(replacement);
}

enum class TrajectoryMessageValidationReason {
  kValid,
  kInvalidValidationPolicy,
  kInvalidCurrentTime,
  kEmptyDuration,
  kUnsupportedOrder,
  kCoefficientLengthMismatch,
  kInvalidDuration,
  kNonFiniteCoefficient,
  kZeroStartTime,
  kInvalidStartTime,
  kStartTimeStale,
  kStartTimeTooFarInFuture,
};

struct TrajectoryMessageValidationResult {
  TrajectoryMessageValidationReason reason;

  bool valid() const { return reason == TrajectoryMessageValidationReason::kValid; }
};

struct TrajectoryMessageValidationPolicy {
  // A planner normally schedules start_time about 0.30 s ahead of the current clock.
  double max_stale_age_sec = 2.0;
  double max_future_start_sec = 1.0;
};

enum class ClockObservation {
  kAccepted,
  kInvalid,
  kRegressed,
};

// Owns the source-time and transport epochs that protect trajectory ingress.  It is
// deliberately ROS-free so queue/clock behavior can be tested without a node.
class TrajectoryIngressEpoch {
public:
  ClockObservation observeClock(double current_time_sec)
  {
    if (!std::isfinite(current_time_sec) || current_time_sec <= 0.0)
      return ClockObservation::kInvalid;
    if (has_clock_ && current_time_sec < last_clock_time_sec_)
      return ClockObservation::kRegressed;
    last_clock_time_sec_ = current_time_sec;
    has_clock_ = true;
    return ClockObservation::kAccepted;
  }

  // A new clock epoch intentionally permits a restarted planner to begin its
  // trajectory IDs from a new sequence. Existing callback closures are invalidated.
  void beginNewClockEpoch(double current_time_sec)
  {
    ++transport_epoch_;
    has_active_trajectory_ = false;
    last_accepted_trajectory_id_ = std::numeric_limits<int>::min();
    last_accepted_start_time_sec_ = -std::numeric_limits<double>::infinity();
    has_stop_epoch_ = false;
    stop_time_sec_ = 0.0;
    has_clock_ = std::isfinite(current_time_sec) && current_time_sec > 0.0;
    last_clock_time_sec_ = has_clock_ ? current_time_sec : 0.0;
  }

  // Stops retain the prior ID floor but invalidate all callback closures from
  // the old subscriber generation.
  void beginStopEpoch(double stop_time_sec)
  {
    ++transport_epoch_;
    has_active_trajectory_ = false;
    has_stop_epoch_ = true;
    stop_time_sec_ = stop_time_sec;
  }

  bool canAcceptTrajectory(std::uint64_t callback_transport_epoch, int trajectory_id,
      double start_time_sec) const
  {
    if (!isCurrentTransportEpoch(callback_transport_epoch) ||
        trajectory_id <= last_accepted_trajectory_id_ ||
        start_time_sec < last_accepted_start_time_sec_ ||
        (has_stop_epoch_ && start_time_sec <= stop_time_sec_)) {
      return false;
    }
    return true;
  }

  void noteAcceptedTrajectory(int trajectory_id, double start_time_sec)
  {
    last_accepted_trajectory_id_ = trajectory_id;
    last_accepted_start_time_sec_ = start_time_sec;
    has_active_trajectory_ = true;
  }

  bool acceptTrajectory(std::uint64_t callback_transport_epoch, int trajectory_id,
      double start_time_sec)
  {
    if (!canAcceptTrajectory(callback_transport_epoch, trajectory_id, start_time_sec))
      return false;
    noteAcceptedTrajectory(trajectory_id, start_time_sec);
    return true;
  }

  void clearActiveTrajectory() { has_active_trajectory_ = false; }

  bool hasActiveTrajectory() const { return has_active_trajectory_; }
  bool hasStopEpoch() const { return has_stop_epoch_; }
  bool isCurrentTransportEpoch(std::uint64_t epoch) const { return epoch == transport_epoch_; }
  std::uint64_t transportEpoch() const { return transport_epoch_; }
  std::uint64_t nextTransportEpoch() const { return transport_epoch_ + 1U; }
  int lastAcceptedTrajectoryId() const { return last_accepted_trajectory_id_; }
  double lastAcceptedStartTimeSec() const { return last_accepted_start_time_sec_; }
  double stopTimeSec() const { return stop_time_sec_; }

private:
  std::uint64_t transport_epoch_ = 1;
  bool has_clock_ = false;
  double last_clock_time_sec_ = 0.0;
  bool has_stop_epoch_ = false;
  double stop_time_sec_ = 0.0;
  int last_accepted_trajectory_id_ = std::numeric_limits<int>::min();
  double last_accepted_start_time_sec_ = -std::numeric_limits<double>::infinity();
  bool has_active_trajectory_ = false;
};

enum class OdometrySampleValidationReason {
  kValid,
  kInvalidPolicy,
  kInvalidCurrentTime,
  kInvalidSourceTime,
  kStaleSourceTime,
  kFutureSourceTime,
  kNonMonotonicSourceTime,
  kFrameMismatch,
  kChildFrameMismatch,
  kInvalidShape,
  kNonFiniteState,
  kNonFiniteCovariance,
  kInvalidQuaternionNorm,
};

struct OdometrySampleValidationResult {
  OdometrySampleValidationReason reason;

  bool valid() const { return reason == OdometrySampleValidationReason::kValid; }
};

struct OdometrySampleValidationPolicy {
  std::string expected_world_frame = "map";
  std::string expected_body_frame = "base_link";
  double max_stale_age_sec = 0.20;
  double max_future_age_sec = 0.02;
  double quaternion_norm_tolerance = 0.05;
};

struct OdometrySampleView {
  double source_time_sec;
  const std::string& frame_id;
  const std::string& child_frame_id;
  const double* position;
  std::size_t position_size;
  const double* orientation_xyzw;
  std::size_t orientation_size;
  const double* twist;
  std::size_t twist_size;
  const double* pose_covariance;
  std::size_t pose_covariance_size;
  const double* twist_covariance;
  std::size_t twist_covariance_size;
};

inline const char* odometrySampleValidationReasonString(OdometrySampleValidationReason reason)
{
  switch (reason) {
    case OdometrySampleValidationReason::kValid:
      return "valid";
    case OdometrySampleValidationReason::kInvalidPolicy:
      return "invalid odometry validation policy";
    case OdometrySampleValidationReason::kInvalidCurrentTime:
      return "current ROS time is zero or non-finite";
    case OdometrySampleValidationReason::kInvalidSourceTime:
      return "odometry source time is zero or non-finite";
    case OdometrySampleValidationReason::kStaleSourceTime:
      return "odometry source time is stale";
    case OdometrySampleValidationReason::kFutureSourceTime:
      return "odometry source time is in the future";
    case OdometrySampleValidationReason::kNonMonotonicSourceTime:
      return "odometry source time is not strictly increasing";
    case OdometrySampleValidationReason::kFrameMismatch:
      return "odometry world frame mismatch";
    case OdometrySampleValidationReason::kChildFrameMismatch:
      return "odometry body frame mismatch";
    case OdometrySampleValidationReason::kInvalidShape:
      return "odometry state or covariance shape is invalid";
    case OdometrySampleValidationReason::kNonFiniteState:
      return "odometry state contains NaN or Inf";
    case OdometrySampleValidationReason::kNonFiniteCovariance:
      return "odometry covariance contains NaN or Inf";
    case OdometrySampleValidationReason::kInvalidQuaternionNorm:
      return "odometry quaternion norm is outside tolerance";
  }
  return "unknown odometry validation failure";
}

inline bool finiteArray(const double* values, std::size_t size)
{
  if (values == nullptr)
    return false;
  for (std::size_t index = 0; index < size; ++index) {
    if (!std::isfinite(values[index]))
      return false;
  }
  return true;
}

inline OdometrySampleValidationResult validateOdometrySample(const OdometrySampleView& sample,
    double current_time_sec, double last_source_time_sec,
    const OdometrySampleValidationPolicy& policy)
{
  if (policy.expected_world_frame.empty() || policy.expected_body_frame.empty() ||
      !std::isfinite(policy.max_stale_age_sec) || policy.max_stale_age_sec < 0.0 ||
      !std::isfinite(policy.max_future_age_sec) || policy.max_future_age_sec < 0.0 ||
      !std::isfinite(policy.quaternion_norm_tolerance) ||
      policy.quaternion_norm_tolerance <= 0.0 || policy.quaternion_norm_tolerance >= 1.0) {
    return {OdometrySampleValidationReason::kInvalidPolicy};
  }
  if (!std::isfinite(current_time_sec) || current_time_sec <= 0.0)
    return {OdometrySampleValidationReason::kInvalidCurrentTime};
  if (!std::isfinite(sample.source_time_sec) || sample.source_time_sec <= 0.0)
    return {OdometrySampleValidationReason::kInvalidSourceTime};
  if (sample.source_time_sec < current_time_sec - policy.max_stale_age_sec)
    return {OdometrySampleValidationReason::kStaleSourceTime};
  if (sample.source_time_sec > current_time_sec + policy.max_future_age_sec)
    return {OdometrySampleValidationReason::kFutureSourceTime};
  if (std::isfinite(last_source_time_sec) && last_source_time_sec > 0.0 &&
      sample.source_time_sec <= last_source_time_sec) {
    return {OdometrySampleValidationReason::kNonMonotonicSourceTime};
  }
  if (sample.frame_id != policy.expected_world_frame)
    return {OdometrySampleValidationReason::kFrameMismatch};
  if (sample.child_frame_id != policy.expected_body_frame)
    return {OdometrySampleValidationReason::kChildFrameMismatch};
  if (sample.position_size != 3U || sample.orientation_size != 4U || sample.twist_size != 6U ||
      sample.pose_covariance_size != 36U || sample.twist_covariance_size != 36U ||
      sample.position == nullptr || sample.orientation_xyzw == nullptr || sample.twist == nullptr ||
      sample.pose_covariance == nullptr || sample.twist_covariance == nullptr) {
    return {OdometrySampleValidationReason::kInvalidShape};
  }
  if (!finiteArray(sample.position, sample.position_size) ||
      !finiteArray(sample.orientation_xyzw, sample.orientation_size) ||
      !finiteArray(sample.twist, sample.twist_size)) {
    return {OdometrySampleValidationReason::kNonFiniteState};
  }
  if (!finiteArray(sample.pose_covariance, sample.pose_covariance_size) ||
      !finiteArray(sample.twist_covariance, sample.twist_covariance_size)) {
    return {OdometrySampleValidationReason::kNonFiniteCovariance};
  }
  double quaternion_norm_squared = 0.0;
  for (std::size_t index = 0; index < sample.orientation_size; ++index)
    quaternion_norm_squared += sample.orientation_xyzw[index] * sample.orientation_xyzw[index];
  const double quaternion_norm = std::sqrt(quaternion_norm_squared);
  if (!std::isfinite(quaternion_norm) ||
      std::abs(quaternion_norm - 1.0) > policy.quaternion_norm_tolerance) {
    return {OdometrySampleValidationReason::kInvalidQuaternionNorm};
  }
  return {OdometrySampleValidationReason::kValid};
}

enum class TrackingConfigurationValidationReason {
  kValid,
  kUnsupportedMode,
  kInvalidProgressParameters,
};

struct TrackingConfigurationValidationResult {
  TrackingConfigurationValidationReason reason;

  bool valid() const { return reason == TrackingConfigurationValidationReason::kValid; }
};

struct TrackingConfiguration {
  std::string mode;
  double full_rate_error;
  double freeze_error;
  double minimum_rate;
};

inline TrackingConfigurationValidationResult validateTrackingConfiguration(
    const TrackingConfiguration& config)
{
  if (config.mode != "px4_native" && config.mode != "legacy_mpc")
    return {TrackingConfigurationValidationReason::kUnsupportedMode};
  if (!std::isfinite(config.full_rate_error) || !std::isfinite(config.freeze_error) ||
      !std::isfinite(config.minimum_rate) || config.full_rate_error < 0.0 ||
      config.freeze_error < 0.0 || config.minimum_rate < 0.0 || config.minimum_rate > 1.0 ||
      !(config.full_rate_error < config.freeze_error)) {
    return {TrackingConfigurationValidationReason::kInvalidProgressParameters};
  }
  return {TrackingConfigurationValidationReason::kValid};
}

inline const char* trackingConfigurationValidationReasonString(
    TrackingConfigurationValidationReason reason)
{
  switch (reason) {
    case TrackingConfigurationValidationReason::kValid:
      return "valid";
    case TrackingConfigurationValidationReason::kUnsupportedMode:
      return "tracking_mode must be px4_native or legacy_mpc";
    case TrackingConfigurationValidationReason::kInvalidProgressParameters:
      return "progress parameters require 0 <= minimum <= 1 and 0 <= full < freeze";
  }
  return "unknown tracking configuration validation failure";
}

template <typename Scalar>
struct TrajectoryMessageView {
  int order;
  double start_time_sec;
  const std::vector<Scalar>& coefficient_x;
  const std::vector<Scalar>& coefficient_y;
  const std::vector<Scalar>& coefficient_z;
  const std::vector<Scalar>& duration;
};

inline const char* trajectoryMessageValidationReasonString(
    TrajectoryMessageValidationReason reason)
{
  switch (reason) {
    case TrajectoryMessageValidationReason::kValid:
      return "valid";
    case TrajectoryMessageValidationReason::kInvalidValidationPolicy:
      return "invalid trajectory validation policy";
    case TrajectoryMessageValidationReason::kInvalidCurrentTime:
      return "current ROS time is zero or non-finite";
    case TrajectoryMessageValidationReason::kEmptyDuration:
      return "trajectory duration is empty";
    case TrajectoryMessageValidationReason::kUnsupportedOrder:
      return "trajectory order is not 7";
    case TrajectoryMessageValidationReason::kCoefficientLengthMismatch:
      return "trajectory coefficient lengths do not equal pieces * 8";
    case TrajectoryMessageValidationReason::kInvalidDuration:
      return "trajectory duration contains a non-positive or non-finite value";
    case TrajectoryMessageValidationReason::kNonFiniteCoefficient:
      return "trajectory coefficients contain NaN or Inf";
    case TrajectoryMessageValidationReason::kZeroStartTime:
      return "trajectory start_time is zero";
    case TrajectoryMessageValidationReason::kInvalidStartTime:
      return "trajectory start_time is non-finite";
    case TrajectoryMessageValidationReason::kStartTimeStale:
      return "trajectory start_time is too old";
    case TrajectoryMessageValidationReason::kStartTimeTooFarInFuture:
      return "trajectory start_time is too far in the future";
  }
  return "unknown trajectory validation failure";
}

template <typename Scalar>
TrajectoryMessageValidationResult validateTrajectoryMessage(const TrajectoryMessageView<Scalar>& msg,
    double current_time_sec, const TrajectoryMessageValidationPolicy& policy)
{
  if (!std::isfinite(policy.max_stale_age_sec) || policy.max_stale_age_sec < 0.0 ||
      !std::isfinite(policy.max_future_start_sec) || policy.max_future_start_sec < 0.0) {
    return {TrajectoryMessageValidationReason::kInvalidValidationPolicy};
  }
  if (!std::isfinite(current_time_sec) || current_time_sec <= 0.0)
    return {TrajectoryMessageValidationReason::kInvalidCurrentTime};
  if (msg.duration.empty())
    return {TrajectoryMessageValidationReason::kEmptyDuration};
  if (msg.order != 7)
    return {TrajectoryMessageValidationReason::kUnsupportedOrder};

  const std::size_t pieces = msg.duration.size();
  if (pieces > std::numeric_limits<std::size_t>::max() / 8 ||
      msg.coefficient_x.size() != pieces * 8 || msg.coefficient_y.size() != pieces * 8 ||
      msg.coefficient_z.size() != pieces * 8) {
    return {TrajectoryMessageValidationReason::kCoefficientLengthMismatch};
  }
  for (const Scalar duration : msg.duration) {
    if (!std::isfinite(static_cast<double>(duration)) || duration <= static_cast<Scalar>(0))
      return {TrajectoryMessageValidationReason::kInvalidDuration};
  }
  for (const std::vector<Scalar>* coefficients :
      {&msg.coefficient_x, &msg.coefficient_y, &msg.coefficient_z}) {
    for (const Scalar coefficient : *coefficients) {
      if (!std::isfinite(static_cast<double>(coefficient)))
        return {TrajectoryMessageValidationReason::kNonFiniteCoefficient};
    }
  }
  if (msg.start_time_sec == 0.0)
    return {TrajectoryMessageValidationReason::kZeroStartTime};
  if (!std::isfinite(msg.start_time_sec))
    return {TrajectoryMessageValidationReason::kInvalidStartTime};
  if (msg.start_time_sec < current_time_sec - policy.max_stale_age_sec)
    return {TrajectoryMessageValidationReason::kStartTimeStale};
  if (msg.start_time_sec > current_time_sec + policy.max_future_start_sec)
    return {TrajectoryMessageValidationReason::kStartTimeTooFarInFuture};
  return {TrajectoryMessageValidationReason::kValid};
}

}  // namespace trajectory_manager
