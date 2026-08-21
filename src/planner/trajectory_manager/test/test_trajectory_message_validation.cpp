#include <gtest/gtest.h>

#include <limits>
#include <array>
#include <vector>

#include <trajectory_manager/trajectory_message_validation.h>
#include <trajectory_manager/tracking_policy.h>

namespace {

using trajectory_manager::TrajectoryMessageValidationPolicy;
using trajectory_manager::TrajectoryMessageValidationReason;
using trajectory_manager::TrajectoryMessageView;
using trajectory_manager::ClockObservation;
using trajectory_manager::TrackingConfiguration;
using trajectory_manager::TrackingConfigurationValidationReason;
using trajectory_manager::TrajectoryIngressEpoch;
using trajectory_manager::OdometrySampleValidationPolicy;
using trajectory_manager::OdometrySampleValidationReason;
using trajectory_manager::OdometrySampleView;
using trajectory_manager::validateTrajectoryMessage;
using trajectory_manager::validateTrackingConfiguration;
using trajectory_manager::validateOdometrySample;

struct TestTrajectoryMessage {
  int order = 7;
  double start_time = 100.0;
  std::vector<float> coefficient_x;
  std::vector<float> coefficient_y;
  std::vector<float> coefficient_z;
  std::vector<float> duration;

  TestTrajectoryMessage()
  {
    duration = {1.0F};
    coefficient_x.assign(8, 0.0F);
    coefficient_y.assign(8, 0.0F);
    coefficient_z.assign(8, 0.0F);
  }

  TrajectoryMessageView<float> view() const
  {
    return {order, start_time, coefficient_x, coefficient_y, coefficient_z, duration};
  }
};

TrajectoryMessageValidationPolicy policy()
{
  TrajectoryMessageValidationPolicy result;
  result.max_stale_age_sec = 2.0;
  result.max_future_start_sec = 1.0;
  return result;
}

TEST(TrajectoryMessageValidation, RejectsEmptyDuration)
{
  TestTrajectoryMessage msg;
  msg.duration.clear();
  EXPECT_EQ(TrajectoryMessageValidationReason::kEmptyDuration,
      validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
}

TEST(TrajectoryMessageValidation, RejectsUnsupportedOrder)
{
  TestTrajectoryMessage msg;
  msg.order = 6;
  EXPECT_EQ(TrajectoryMessageValidationReason::kUnsupportedOrder,
      validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
}

TEST(TrajectoryMessageValidation, RejectsMismatchedCoefficientLengths)
{
  TestTrajectoryMessage msg;
  msg.coefficient_y.pop_back();
  EXPECT_EQ(TrajectoryMessageValidationReason::kCoefficientLengthMismatch,
      validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
}

TEST(TrajectoryMessageValidation, RejectsNonPositiveAndNonFiniteDurations)
{
  for (const float invalid : {0.0F, -1.0F, std::numeric_limits<float>::quiet_NaN(),
           std::numeric_limits<float>::infinity()}) {
    TestTrajectoryMessage msg;
    msg.duration[0] = invalid;
    EXPECT_EQ(TrajectoryMessageValidationReason::kInvalidDuration,
        validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
  }
}

TEST(TrajectoryMessageValidation, RejectsNonFiniteCoefficientsOnEveryAxis)
{
  for (std::vector<float> TestTrajectoryMessage::*axis : {
           &TestTrajectoryMessage::coefficient_x, &TestTrajectoryMessage::coefficient_y,
           &TestTrajectoryMessage::coefficient_z}) {
    TestTrajectoryMessage msg;
    (msg.*axis)[3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(TrajectoryMessageValidationReason::kNonFiniteCoefficient,
        validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);

    (msg.*axis)[3] = std::numeric_limits<float>::infinity();
    EXPECT_EQ(TrajectoryMessageValidationReason::kNonFiniteCoefficient,
        validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
  }
}

TEST(TrajectoryMessageValidation, RejectsZeroStaleAndTooFutureStartTimes)
{
  TestTrajectoryMessage msg;
  msg.start_time = 0.0;
  EXPECT_EQ(TrajectoryMessageValidationReason::kZeroStartTime,
      validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);

  msg.start_time = 97.99;
  EXPECT_EQ(TrajectoryMessageValidationReason::kStartTimeStale,
      validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);

  msg.start_time = 101.01;
  EXPECT_EQ(TrajectoryMessageValidationReason::kStartTimeTooFarInFuture,
      validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
}

TEST(TrajectoryMessageValidation, AcceptsCurrentAndPrepareWindowStartTimes)
{
  for (const double valid_start : {98.0, 100.0, 100.30, 101.0}) {
    TestTrajectoryMessage msg;
    msg.start_time = valid_start;
    EXPECT_EQ(TrajectoryMessageValidationReason::kValid,
        validateTrajectoryMessage(msg.view(), 100.0, policy()).reason);
  }
}

TEST(TrajectoryIngressEpoch, ClockRegressionStopsOldExecutionAndStartsANewEpoch)
{
  TrajectoryIngressEpoch ingress;
  ASSERT_EQ(ClockObservation::kAccepted, ingress.observeClock(100.0));
  const std::uint64_t old_epoch = ingress.transportEpoch();
  ASSERT_TRUE(ingress.acceptTrajectory(old_epoch, 7, 100.2));
  ASSERT_TRUE(ingress.hasActiveTrajectory());

  EXPECT_EQ(ClockObservation::kRegressed, ingress.observeClock(10.0));
  ingress.beginNewClockEpoch(10.0);

  EXPECT_FALSE(ingress.hasActiveTrajectory());
  EXPECT_FALSE(ingress.isCurrentTransportEpoch(old_epoch));
  EXPECT_EQ(std::numeric_limits<int>::min(), ingress.lastAcceptedTrajectoryId());
  EXPECT_FALSE(ingress.hasStopEpoch());

  const std::uint64_t new_epoch = ingress.transportEpoch();
  EXPECT_TRUE(ingress.acceptTrajectory(new_epoch, 1, 10.3));
  EXPECT_TRUE(ingress.hasActiveTrajectory());
}

TEST(TrajectoryIngressEpoch, StopTransportBarrierRejectsDelayedHighIdOldCallback)
{
  TrajectoryIngressEpoch ingress;
  ASSERT_EQ(ClockObservation::kAccepted, ingress.observeClock(100.0));
  const std::uint64_t old_epoch = ingress.transportEpoch();
  ASSERT_TRUE(ingress.acceptTrajectory(old_epoch, 7, 100.1));

  ingress.beginStopEpoch(100.2);
  const std::uint64_t new_epoch = ingress.transportEpoch();
  EXPECT_FALSE(ingress.hasActiveTrajectory());
  EXPECT_FALSE(ingress.acceptTrajectory(old_epoch, 99, 100.3));
  EXPECT_FALSE(ingress.acceptTrajectory(new_epoch, 99, 100.1));
  EXPECT_FALSE(ingress.acceptTrajectory(new_epoch, 7, 100.3));
  EXPECT_TRUE(ingress.acceptTrajectory(new_epoch, 8, 100.3));
}

TEST(TrajectoryIngressEpoch, RejectsSourceTimeRegressionUntilClockEpochReset)
{
  TrajectoryIngressEpoch ingress;
  ASSERT_EQ(ClockObservation::kAccepted, ingress.observeClock(100.0));
  const std::uint64_t first_epoch = ingress.transportEpoch();
  ASSERT_TRUE(ingress.acceptTrajectory(first_epoch, 7, 100.4));

  // A higher ID must not make an older source start time valid in the same
  // clock epoch.
  EXPECT_FALSE(ingress.acceptTrajectory(first_epoch, 8, 100.3));
  EXPECT_TRUE(ingress.acceptTrajectory(first_epoch, 8, 100.4));

  ingress.beginNewClockEpoch(10.0);
  const std::uint64_t reset_epoch = ingress.transportEpoch();
  EXPECT_TRUE(ingress.acceptTrajectory(reset_epoch, 1, 10.3));
}

struct FakeSubscriber {
  explicit FakeSubscriber(std::vector<std::string>* events_in = nullptr, int generation_in = 0)
    : events(events_in), generation(generation_in)
  {
  }

  void shutdown()
  {
    if (events)
      events->push_back("shutdown-old");
  }

  std::vector<std::string>* events;
  int generation;
};

TEST(TrajectorySubscriberSwap, CreatesReplacementBeforeClosingOldSubscription)
{
  std::vector<std::string> events;
  FakeSubscriber active(&events, 1);

  trajectory_manager::overlapSwapSubscriber(active, [&events]() {
    events.push_back("create-new");
    return FakeSubscriber(&events, 2);
  }, [&events]() {
    events.push_back("activate-epoch");
  });

  ASSERT_EQ(3U, events.size());
  EXPECT_EQ("create-new", events[0]);
  EXPECT_EQ("activate-epoch", events[1]);
  EXPECT_EQ("shutdown-old", events[2]);
  EXPECT_EQ(2, active.generation);
}

TEST(TrajectorySubscriberSwap, PreparedCallbackBecomesCurrentWithoutAReceptionGap)
{
  TrajectoryIngressEpoch ingress;
  ASSERT_EQ(ClockObservation::kAccepted, ingress.observeClock(100.0));
  const std::uint64_t old_epoch = ingress.transportEpoch();
  std::uint64_t prepared_epoch = old_epoch;
  FakeSubscriber active(nullptr, 1);

  trajectory_manager::overlapSwapSubscriber(active, [&]() {
    prepared_epoch = ingress.nextTransportEpoch();
    return FakeSubscriber(nullptr, 2);
  }, [&]() {
    ingress.beginStopEpoch(100.2);
  });

  EXPECT_FALSE(ingress.isCurrentTransportEpoch(old_epoch));
  EXPECT_TRUE(ingress.isCurrentTransportEpoch(prepared_epoch));
  EXPECT_FALSE(ingress.acceptTrajectory(old_epoch, 99, 100.3));
  EXPECT_TRUE(ingress.acceptTrajectory(prepared_epoch, 8, 100.3));
}

struct TestOdometrySample {
  double stamp = 100.0;
  std::string frame_id = "map";
  std::string child_frame_id = "base_link";
  std::array<double, 3> position{{1.0, 2.0, 1.0}};
  std::array<double, 4> orientation_xyzw{{0.0, 0.0, 0.0, 1.0}};
  std::array<double, 6> twist{{0.1, 0.0, 0.0, 0.0, 0.0, 0.1}};
  std::array<double, 36> pose_covariance{{}};
  std::array<double, 36> twist_covariance{{}};

  OdometrySampleView view() const
  {
    return {stamp, frame_id, child_frame_id, position.data(), position.size(),
        orientation_xyzw.data(), orientation_xyzw.size(), twist.data(), twist.size(),
        pose_covariance.data(), pose_covariance.size(), twist_covariance.data(),
        twist_covariance.size()};
  }
};

OdometrySampleValidationPolicy odometryPolicy()
{
  OdometrySampleValidationPolicy result;
  result.expected_world_frame = "map";
  result.expected_body_frame = "base_link";
  result.max_stale_age_sec = 0.20;
  result.max_future_age_sec = 0.02;
  result.quaternion_norm_tolerance = 0.05;
  return result;
}

TEST(OdometrySampleValidation, AcceptsFreshFiniteExpectedFrameSample)
{
  TestOdometrySample odom;
  EXPECT_EQ(OdometrySampleValidationReason::kValid,
      validateOdometrySample(odom.view(), 100.1, 99.9, odometryPolicy()).reason);
}

TEST(OdometrySampleValidation, RejectsWrongFramesAndNonMonotonicSourceTime)
{
  TestOdometrySample odom;
  odom.frame_id = "odom";
  EXPECT_EQ(OdometrySampleValidationReason::kFrameMismatch,
      validateOdometrySample(odom.view(), 100.1, 99.9, odometryPolicy()).reason);
  odom.frame_id = "map";
  odom.child_frame_id = "camera";
  EXPECT_EQ(OdometrySampleValidationReason::kChildFrameMismatch,
      validateOdometrySample(odom.view(), 100.1, 99.9, odometryPolicy()).reason);
  odom.child_frame_id = "base_link";
  EXPECT_EQ(OdometrySampleValidationReason::kNonMonotonicSourceTime,
      validateOdometrySample(odom.view(), 100.1, 100.0, odometryPolicy()).reason);
}

TEST(OdometrySampleValidation, RejectsZeroStaleFutureAndNonFiniteSourceTime)
{
  TestOdometrySample odom;
  odom.stamp = 0.0;
  EXPECT_EQ(OdometrySampleValidationReason::kInvalidSourceTime,
      validateOdometrySample(odom.view(), 100.0, 0.0, odometryPolicy()).reason);
  odom.stamp = 99.79;
  EXPECT_EQ(OdometrySampleValidationReason::kStaleSourceTime,
      validateOdometrySample(odom.view(), 100.0, 0.0, odometryPolicy()).reason);
  odom.stamp = 100.03;
  EXPECT_EQ(OdometrySampleValidationReason::kFutureSourceTime,
      validateOdometrySample(odom.view(), 100.0, 0.0, odometryPolicy()).reason);
  odom.stamp = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(OdometrySampleValidationReason::kInvalidSourceTime,
      validateOdometrySample(odom.view(), 100.0, 0.0, odometryPolicy()).reason);
}

TEST(OdometrySampleValidation, RejectsNonFiniteStateCovarianceAndBadQuaternion)
{
  TestOdometrySample odom;
  odom.position[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(OdometrySampleValidationReason::kNonFiniteState,
      validateOdometrySample(odom.view(), 100.1, 0.0, odometryPolicy()).reason);
  odom.position[1] = 2.0;
  odom.twist[5] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(OdometrySampleValidationReason::kNonFiniteState,
      validateOdometrySample(odom.view(), 100.1, 0.0, odometryPolicy()).reason);
  odom.twist[5] = 0.1;
  odom.pose_covariance[17] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(OdometrySampleValidationReason::kNonFiniteCovariance,
      validateOdometrySample(odom.view(), 100.1, 0.0, odometryPolicy()).reason);
  odom.pose_covariance[17] = 0.0;
  odom.orientation_xyzw[3] = 0.5;
  EXPECT_EQ(OdometrySampleValidationReason::kInvalidQuaternionNorm,
      validateOdometrySample(odom.view(), 100.1, 0.0, odometryPolicy()).reason);
}

TEST(TrajectoryTrackingConfiguration, RejectsUnsafeModesAndProgressParameters)
{
  TrackingConfiguration config;
  config.mode = "unsupported";
  config.full_rate_error = 0.12;
  config.freeze_error = 0.25;
  config.minimum_rate = 0.25;
  EXPECT_EQ(TrackingConfigurationValidationReason::kUnsupportedMode,
      validateTrackingConfiguration(config).reason);

  config.mode = "px4_native";
  config.full_rate_error = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(TrackingConfigurationValidationReason::kInvalidProgressParameters,
      validateTrackingConfiguration(config).reason);

  config.full_rate_error = 0.25;
  config.freeze_error = 0.25;
  EXPECT_EQ(TrackingConfigurationValidationReason::kInvalidProgressParameters,
      validateTrackingConfiguration(config).reason);

  config.full_rate_error = 0.12;
  config.freeze_error = 0.25;
  config.minimum_rate = 1.01;
  EXPECT_EQ(TrackingConfigurationValidationReason::kInvalidProgressParameters,
      validateTrackingConfiguration(config).reason);
}

TEST(TrajectoryTrackingConfiguration, AcceptsSupportedModesAndBoundsProgressRate)
{
  for (const std::string mode : {"px4_native", "legacy_mpc"}) {
    TrackingConfiguration config;
    config.mode = mode;
    config.full_rate_error = 0.12;
    config.freeze_error = 0.25;
    config.minimum_rate = 0.25;
    EXPECT_EQ(TrackingConfigurationValidationReason::kValid,
        validateTrackingConfiguration(config).reason);
    for (const double error : {0.0, 0.12, 0.185, 0.25, 10.0}) {
      const double rate = trajectory_manager::progressRate(
          error, config.full_rate_error, config.freeze_error, config.minimum_rate);
      EXPECT_GE(rate, 0.0);
      EXPECT_LE(rate, 1.0);
    }
  }
}

}  // namespace
