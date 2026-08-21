#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include <exploration_manager/robust_navigation_policy.h>

using apexnav_planner::ReplanSettleGate;
using apexnav_planner::shouldAcceptPlannerTrigger;
using apexnav_planner::shouldReuseLockedTarget;
using apexnav_planner::trajectoryStartIsContinuous;

TEST(RobustNavigationPolicy, ReusesTargetAcrossOrdinaryLocalReplans)
{
  EXPECT_TRUE(shouldReuseLockedTarget(true, 2.0, false, 4.0, 10.0, 1, 3, 0.35));
  EXPECT_FALSE(shouldReuseLockedTarget(true, 2.0, true, 4.0, 10.0, 1, 3, 0.35));
  EXPECT_FALSE(shouldReuseLockedTarget(true, 2.0, false, 11.0, 10.0, 1, 3, 0.35));
  EXPECT_FALSE(shouldReuseLockedTarget(true, 0.20, false, 4.0, 10.0, 1, 3, 0.35));
  EXPECT_FALSE(shouldReuseLockedTarget(true, 2.0, false, 4.0, 10.0, 3, 3, 0.35));
}

TEST(RobustNavigationPolicy, RequiresAStableBrakeWindowBeforeReplanning)
{
  ReplanSettleGate gate;
  EXPECT_FALSE(gate.update(1.00, 0.12, 0.05, 0.50));
  EXPECT_FALSE(gate.update(1.10, 0.04, 0.05, 0.50));
  EXPECT_FALSE(gate.update(1.55, 0.04, 0.05, 0.50));
  EXPECT_TRUE(gate.update(1.61, 0.04, 0.05, 0.50));
  EXPECT_FALSE(gate.update(1.70, 0.08, 0.05, 0.50));
}

TEST(RobustNavigationPolicy, RejectsDiscontinuousTrajectoryStart)
{
  EXPECT_TRUE(trajectoryStartIsContinuous(0.099, 0.10));
  EXPECT_FALSE(trajectoryStartIsContinuous(0.101, 0.10));
}

TEST(RobustNavigationPolicy, RejectsPlannerTriggerUntilNavigationIsEnabled)
{
  EXPECT_FALSE(shouldAcceptPlannerTrigger(true, false));
  EXPECT_FALSE(shouldAcceptPlannerTrigger(false, true));
  EXPECT_TRUE(shouldAcceptPlannerTrigger(true, true));
}

TEST(RobustNavigationPolicy, RejectsStaleFutureAndReplayedSourceStamps)
{
  EXPECT_TRUE(apexnav_planner::sourceStampIsAcceptable(10.0, 9.8, 9.7, 0.5, 0.05));
  EXPECT_FALSE(apexnav_planner::sourceStampIsAcceptable(10.0, 0.0, 0.0, 0.5, 0.05));
  EXPECT_FALSE(apexnav_planner::sourceStampIsAcceptable(10.0, 9.4, 0.0, 0.5, 0.05));
  EXPECT_FALSE(apexnav_planner::sourceStampIsAcceptable(10.0, 10.1, 0.0, 0.5, 0.05));
  EXPECT_FALSE(apexnav_planner::sourceStampIsAcceptable(10.0, 9.8, 9.8, 0.5, 0.05));
}

TEST(RobustNavigationPolicy, ValidatesPlannerScalarIngress)
{
  EXPECT_TRUE(apexnav_planner::confidenceThresholdIsValid(0.5));
  EXPECT_FALSE(apexnav_planner::confidenceThresholdIsValid(-0.1));
  EXPECT_FALSE(apexnav_planner::confidenceThresholdIsValid(1.1));
  EXPECT_FALSE(apexnav_planner::confidenceThresholdIsValid(
      std::numeric_limits<double>::quiet_NaN()));

  EXPECT_TRUE(apexnav_planner::trajectoryProgressIsValid(0.02, 0.0, 3.0, false, 0.25));
  EXPECT_FALSE(apexnav_planner::trajectoryProgressIsValid(1.0, 0.0, 3.0, false, 0.25));
  EXPECT_TRUE(apexnav_planner::trajectoryProgressIsValid(1.0, 0.9, 3.0, true, 0.25));
  EXPECT_FALSE(apexnav_planner::trajectoryProgressIsValid(0.8, 0.9, 3.0, true, 0.25));
  EXPECT_FALSE(apexnav_planner::trajectoryProgressIsValid(3.1, 0.9, 3.0, true, 0.25));
  EXPECT_FALSE(apexnav_planner::trajectoryProgressIsValid(
      std::numeric_limits<double>::infinity(), 0.9, 3.0, true, 0.25));
}

TEST(RobustNavigationPolicy, PreservesTargetForAMinimumFailureHysteresis)
{
  // With one expensive failed plan per second, retain the same target for the
  // first two failures and release it on the third at two seconds. This stays
  // below a five-second supervisor phase budget without reverting to 0.75 s
  // target churn based on the nominal 0.25 s retry timer.
  EXPECT_TRUE(shouldReuseLockedTarget(
      true, 2.0, false, 0.0, 10.0, 1, 3, 0.35, 0.0, 2.0));
  EXPECT_TRUE(shouldReuseLockedTarget(
      true, 2.0, false, 1.0, 10.0, 2, 3, 0.35, 1.0, 2.0));
  EXPECT_FALSE(shouldReuseLockedTarget(
      true, 2.0, false, 2.0, 10.0, 3, 3, 0.35, 2.0, 2.0));
  EXPECT_TRUE(shouldReuseLockedTarget(
      true, 2.0, false, 8.0, 10.0, 2, 3, 0.35, 8.0, 2.0));

  // Safety/reached gates remain immediate regardless of the larger failure budget.
  EXPECT_FALSE(shouldReuseLockedTarget(
      true, 2.0, true, 0.2, 10.0, 0, 3, 0.35, 0.0, 2.0));
  EXPECT_FALSE(shouldReuseLockedTarget(
      true, 0.20, false, 0.2, 10.0, 0, 3, 0.35, 0.0, 2.0));
}

TEST(RobustNavigationPolicy, RejectsNonFiniteOrMalformedOdometryQuaternions)
{
  EXPECT_TRUE(apexnav_planner::quaternionIsFiniteAndNormalized(0.0, 0.0, 0.0, 1.0, 0.05));
  EXPECT_FALSE(apexnav_planner::quaternionIsFiniteAndNormalized(0.0, 0.0, 0.0, 0.0, 0.05));
  EXPECT_FALSE(apexnav_planner::quaternionIsFiniteAndNormalized(0.0, 0.0, 0.0, 1.2, 0.05));
  EXPECT_FALSE(apexnav_planner::quaternionIsFiniteAndNormalized(
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0, 0.05));
}

TEST(RobustNavigationPolicy, SamplesFinalTrajectoryAtBoundedSpaceAndTimeIntervals)
{
  const auto sampling = apexnav_planner::makeFinalTrajectorySampling(1.0, 0.40, 0.05, 0.10);

  ASSERT_TRUE(sampling.valid);
  ASSERT_FALSE(sampling.times.empty());
  EXPECT_DOUBLE_EQ(0.0, sampling.times.front());
  EXPECT_DOUBLE_EQ(1.0, sampling.times.back());
  EXPECT_LE(sampling.time_step, 0.10);
  EXPECT_LE(sampling.time_step * 0.40, 0.05);
  for (size_t i = 1; i < sampling.times.size(); ++i)
    EXPECT_LE(sampling.times[i] - sampling.times[i - 1], sampling.time_step + 1e-12);

  EXPECT_FALSE(apexnav_planner::makeFinalTrajectorySampling(
      std::numeric_limits<double>::quiet_NaN(), 0.40, 0.05, 0.10).valid);
  EXPECT_FALSE(apexnav_planner::makeFinalTrajectorySampling(1.0, 0.0, 0.05, 0.10).valid);
}

TEST(RobustNavigationPolicy, KeepsLastValidYawDuringLowSpeedSamples)
{
  const double previous_yaw = 0.73;
  EXPECT_NEAR(M_PI_2, apexnav_planner::yawForTrajectorySample(0.0, 0.25, previous_yaw), 1e-12);
  EXPECT_DOUBLE_EQ(previous_yaw,
      apexnav_planner::yawForTrajectorySample(0.01, 0.0, previous_yaw));
  EXPECT_DOUBLE_EQ(previous_yaw, apexnav_planner::yawForTrajectorySample(
      std::numeric_limits<double>::quiet_NaN(), 0.0, previous_yaw));
}

TEST(RobustNavigationPolicy, MapsTerminalResultsWithoutAmbiguity)
{
  EXPECT_EQ(apexnav_planner::TerminalNavigationResult::MISSION_FAILED,
      apexnav_planner::terminalNavigationResult(3));
  EXPECT_EQ(apexnav_planner::TerminalNavigationResult::MISSION_SUCCEEDED,
      apexnav_planner::terminalNavigationResult(4));
  EXPECT_EQ(apexnav_planner::TerminalNavigationResult::NOT_TERMINAL,
      apexnav_planner::terminalNavigationResult(0));
}

TEST(RobustNavigationPolicy, RealTrajectoryFsmHasNoInitialPoseControlBypass)
{
  std::ifstream source(EXPLORATION_FSM_TRAJ_SOURCE);
  ASSERT_TRUE(source.good());
  std::stringstream contents;
  contents << source.rdbuf();
  const std::string fsm_source = contents.str();
  EXPECT_EQ(std::string::npos, fsm_source.find("/initialpose"));
  EXPECT_EQ(std::string::npos, fsm_source.find("goalCallback"));

  const std::size_t structure_validation =
      fsm_source.find("validateTrajectoryStructure(*info, \"before trajectory evaluation\"");
  const std::size_t first_trajectory_evaluation = fsm_source.find("info->traj.getPos(0.0)");
  const std::size_t limit_enforcement = fsm_source.find("enforceTrajectoryLimits(*info)");
  const std::size_t post_limit_validation =
      fsm_source.find("validateTrajectoryStructure(*info, \"after trajectory limiting\"");
  const std::size_t final_validation = fsm_source.find("validateFinalTrajectory(*info)");
  const std::size_t newest_assignment = fsm_source.find("fd_->newest_traj_ =");
  ASSERT_NE(std::string::npos, structure_validation);
  ASSERT_NE(std::string::npos, first_trajectory_evaluation);
  ASSERT_NE(std::string::npos, limit_enforcement);
  ASSERT_NE(std::string::npos, post_limit_validation);
  ASSERT_NE(std::string::npos, final_validation);
  ASSERT_NE(std::string::npos, newest_assignment);
  EXPECT_LT(structure_validation, first_trajectory_evaluation);
  EXPECT_LT(limit_enforcement, post_limit_validation);
  EXPECT_LT(post_limit_validation, final_validation);
  EXPECT_LT(final_validation, newest_assignment);
}

TEST(RobustNavigationPolicy, RejectsMalformedTrajectoryPieceLayoutsBeforeEvaluation)
{
  using apexnav_planner::TrajectoryPieceLayout;
  std::vector<TrajectoryPieceLayout> pieces;
  EXPECT_FALSE(apexnav_planner::hasValidTrajectoryPieceLayout(pieces));

  pieces.push_back({std::numeric_limits<double>::infinity(), std::vector<double>(24, 0.0)});
  EXPECT_FALSE(apexnav_planner::hasValidTrajectoryPieceLayout(pieces));

  pieces[0].duration = 0.20;
  pieces.push_back({0.01, std::vector<double>(24, 0.0)});
  pieces[1].coefficients[9] = std::numeric_limits<double>::quiet_NaN();
  pieces.push_back({0.20, std::vector<double>(24, 0.0)});
  EXPECT_FALSE(apexnav_planner::hasValidTrajectoryPieceLayout(pieces));
}

TEST(RobustNavigationPolicy, DoesNotTreatALocalPrefixAsObjectArrival)
{
  // A local target can be close while the globally selected object approach
  // pose remains far away. Only the latter may complete the mission.
  const double local_target_distance = 0.10;
  EXPECT_LT(local_target_distance, 0.25);
  EXPECT_FALSE(apexnav_planner::hasReachedFinalObjectApproachGoal(
      0.0, 0.0, 1.0, 0.0, 0.25));
  EXPECT_TRUE(apexnav_planner::hasReachedFinalObjectApproachGoal(
      0.0, 0.0, 0.20, 0.0, 0.25));
}

TEST(RobustNavigationPolicy, AdaptivelySubdividesCurvedTrajectoryUsingActualPositionSteps)
{
  const auto sampling = apexnav_planner::makeAdaptiveFinalTrajectorySamples(
      std::vector<double>{1.0},
      std::vector<double>{0.20 * M_PI},
      [](std::size_t, double t) {
        return apexnav_planner::TrajectorySamplePoint{0.0, 0.20 * std::sin(M_PI * t), 0.0};
      },
      0.05, 0.05, 1e-6, 32, 100000);

  ASSERT_TRUE(sampling.valid);
  ASSERT_GE(sampling.samples.size(), 3u);
  EXPECT_DOUBLE_EQ(0.0, sampling.samples.front().global_time);
  EXPECT_DOUBLE_EQ(1.0, sampling.samples.back().global_time);
  for (std::size_t i = 1; i < sampling.samples.size(); ++i) {
    EXPECT_LE(sampling.samples[i].global_time - sampling.samples[i - 1].global_time,
        0.05 + 1e-12);
    EXPECT_LE(apexnav_planner::trajectorySampleDistance(
                  sampling.samples[i].position, sampling.samples[i - 1].position),
        0.05 + 1e-12);
  }
}

TEST(RobustNavigationPolicy, IncludesBothEndpointsOfEveryPieceIncludingAShortMiddlePiece)
{
  const std::vector<double> durations{0.10, 0.005, 0.10};
  const auto sampling = apexnav_planner::makeAdaptiveFinalTrajectorySamples(
      durations,
      std::vector<double>{1.0, 1.0, 1.0},
      [](std::size_t piece_index, double local_time) {
        const double offsets[] = {0.0, 0.10, 0.105};
        return apexnav_planner::TrajectorySamplePoint{
            offsets[piece_index] + local_time, 0.0, 0.0};
      },
      0.05, 0.05, 1e-6, 32, 100000);

  ASSERT_TRUE(sampling.valid);
  bool saw_short_start = false;
  bool saw_short_end = false;
  for (const auto& sample : sampling.samples) {
    if (sample.piece_index == 1 && sample.local_time == 0.0)
      saw_short_start = true;
    if (sample.piece_index == 1 && sample.local_time == durations[1])
      saw_short_end = true;
  }
  EXPECT_TRUE(saw_short_start);
  EXPECT_TRUE(saw_short_end);
}

TEST(RobustNavigationPolicy, DoesNotAliasAFinitePolynomialSpikeBetweenDyadicSamples)
{
  // This degree-six polynomial, representable by the degree-seven trajectory
  // container, has zero position and zero velocity at 0, 0.02, and 0.04 s.
  // Endpoint/midpoint sampling therefore aliases the 0.18 m excursion at
  // 0.01 s unless the sampler is constrained by a derivative bound.
  constexpr double duration = 0.04;
  constexpr double gain = 2.0e10;
  const auto hidden_spike = [](double t) {
    const double q = t * (t - 0.02) * (t - 0.04);
    return gain * q * q;
  };
  // y(t) = gain * (t^3 - 0.06 t^2 + 0.0008 t)^2.
  std::vector<std::vector<double>> power_coefficients(2, std::vector<double>(8, 0.0));
  power_coefficients[1][2] = gain * 6.4e-7;
  power_coefficients[1][3] = gain * -9.6e-5;
  power_coefficients[1][4] = gain * 5.2e-3;
  power_coefficients[1][5] = gain * -0.12;
  power_coefficients[1][6] = gain;
  double derivative_bound = 0.0;
  ASSERT_TRUE(apexnav_planner::conservativeBernsteinDerivativeNormBound(
      power_coefficients, duration, &derivative_bound));
  EXPECT_GT(derivative_bound, 0.0);
  const auto sampling = apexnav_planner::makeAdaptiveFinalTrajectorySamples(
      std::vector<double>{duration},
      std::vector<double>{derivative_bound},
      [&](std::size_t, double t) {
        return apexnav_planner::TrajectorySamplePoint{0.0, hidden_spike(t), 0.0};
      },
      0.05, 0.05, 1e-6, 32, 100000);

  ASSERT_TRUE(sampling.valid);
  double largest_sampled_excursion = 0.0;
  for (const auto& sample : sampling.samples)
    largest_sampled_excursion = std::max(largest_sampled_excursion,
        std::abs(sample.position.y));
  EXPECT_GT(largest_sampled_excursion, 0.10);

  power_coefficients[1][3] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(apexnav_planner::conservativeBernsteinDerivativeNormBound(
      power_coefficients, duration, &derivative_bound));
}

TEST(RobustNavigationPolicy, ConservativelyBoundsYawRateBetweenLimitSamples)
{
  // x=t, y=t^2 gives yaw_rate = 2/(1+4t^2+1e-3), whose maximum is at t=0.
  std::vector<std::vector<double>> power_coefficients(2, std::vector<double>(8, 0.0));
  power_coefficients[0][1] = 1.0;
  power_coefficients[1][2] = 1.0;
  double yaw_rate_bound = 0.0;
  ASSERT_TRUE(apexnav_planner::conservativeBernsteinPlanarYawRateBound(
      power_coefficients, 1.0, 1e-3, 8, &yaw_rate_bound));
  EXPECT_GE(yaw_rate_bound, 2.0 / 1.001);
  EXPECT_LT(yaw_rate_bound, 2.01);

  power_coefficients[0][1] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(apexnav_planner::conservativeBernsteinPlanarYawRateBound(
      power_coefficients, 1.0, 1e-3, 8, &yaw_rate_bound));
}
