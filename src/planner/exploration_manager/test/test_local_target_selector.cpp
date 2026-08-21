#include <gtest/gtest.h>

#include <exploration_manager/local_target_selector.h>

using apexnav_planner::LocalTargetSelection;
using apexnav_planner::selectFootprintSafeLocalTarget;

TEST(LocalTargetSelector, DoesNotJumpAcrossAnUnsafePathPrefix)
{
  const Eigen::Vector2d current(0.0, 0.0);
  const std::vector<Eigen::Vector2d> path = {
    current, { 0.25, 0.0 }, { 0.25, 0.0 }, { 0.50, 0.0 }, { 0.50, 0.0 },
    { 0.75, 0.0 }, { 0.75, 0.0 }, { 1.00, 0.0 }, { 1.00, 0.0 },
    { 1.25, 0.0 }, { 2.50, 0.0 }
  };
  const auto collision = [](const Eigen::Vector2d& pos, double) {
    return pos.x() > 0.20 && pos.x() < 2.0;
  };

  const LocalTargetSelection result =
      selectFootprintSafeLocalTarget(current, path, 1.0, 0.30, collision);

  EXPECT_FALSE(result.valid);
  EXPECT_NEAR(result.position.x(), 0.0, 1e-9);
  EXPECT_NEAR(result.position.y(), 0.0, 1e-9);
}

TEST(LocalTargetSelector, UsesFarthestSafePointWithinLocalHorizon)
{
  const Eigen::Vector2d current(0.0, 0.0);
  const std::vector<Eigen::Vector2d> path = {
    current, { 0.25, 0.0 }, { 0.50, 0.0 }, { 0.75, 0.0 }, { 1.00, 0.0 },
    { 1.25, 0.0 }
  };
  const auto collision = [](const Eigen::Vector2d& pos, double) {
    return pos.x() >= 1.0;
  };

  const LocalTargetSelection result =
      selectFootprintSafeLocalTarget(current, path, 1.0, 0.30, collision);

  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.position.x(), 0.75, 1e-9);
  EXPECT_NEAR(result.position.y(), 0.0, 1e-9);
  EXPECT_NEAR(result.path_distance, 0.75, 1e-9);
}

TEST(LocalTargetSelector, HandlesDuplicatedPathPointsWithoutUndefinedYaw)
{
  const Eigen::Vector2d current(0.0, 0.0);
  const std::vector<Eigen::Vector2d> path = {
    current, current, { 0.25, 0.0 }, { 0.25, 0.0 }, { 0.50, 0.0 }
  };
  const auto collision = [](const Eigen::Vector2d&, double yaw) {
    return !std::isfinite(yaw);
  };

  const LocalTargetSelection result =
      selectFootprintSafeLocalTarget(current, path, 0.50, 0.20, collision);

  ASSERT_TRUE(result.valid);
  EXPECT_TRUE(std::isfinite(result.yaw));
  EXPECT_NEAR(result.position.x(), 0.50, 1e-9);
}

TEST(LocalTargetSelector, AcceptsShortSafeProgressInsideLaunchClearBubble)
{
  const Eigen::Vector2d current(0.0, 0.0);
  const std::vector<Eigen::Vector2d> path = {
    current, { 0.10, 0.0 }, { 0.20, 0.0 }, { 0.30, 0.0 }
  };
  const auto collision = [](const Eigen::Vector2d& pos, double) {
    return pos.x() > 0.20;
  };

  const LocalTargetSelection result =
      selectFootprintSafeLocalTarget(current, path, 1.0, 0.10, collision);

  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.position.x(), 0.20, 1e-9);
  EXPECT_NEAR(result.path_distance, 0.20, 1e-9);
}
