#include <gtest/gtest.h>

#include <limits>

#include <trajectory_manager/tracking_policy.h>

using trajectory_manager::progressRate;

TEST(TrajectoryTrackingPolicy, RunsNormallyForSmallError)
{
  EXPECT_DOUBLE_EQ(progressRate(0.10, 0.12, 0.25, 0.25), 1.0);
}

TEST(TrajectoryTrackingPolicy, SlowsProgressivelyThenFreezes)
{
  const double middle = progressRate(0.185, 0.12, 0.25, 0.25);
  EXPECT_GT(middle, 0.25);
  EXPECT_LT(middle, 1.0);
  EXPECT_DOUBLE_EQ(progressRate(0.25, 0.12, 0.25, 0.25), 0.0);
  EXPECT_DOUBLE_EQ(progressRate(0.40, 0.12, 0.25, 0.25), 0.0);
}

TEST(TrajectoryTrackingPolicy, FailsClosedForNonFiniteTrackingError)
{
  EXPECT_DOUBLE_EQ(progressRate(std::numeric_limits<double>::quiet_NaN(), 0.12, 0.25, 0.25), 0.0);
  EXPECT_DOUBLE_EQ(progressRate(std::numeric_limits<double>::infinity(), 0.12, 0.25, 0.25), 0.0);
}
