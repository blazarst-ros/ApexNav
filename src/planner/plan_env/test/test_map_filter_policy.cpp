#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <plan_env/map_filter_policy.h>
#include <plan_env/semantic_cloud_contract.h>

namespace {

using apexnav_planner::RaycastEpoch;
using apexnav_planner::SelfFilterFootprint;
using apexnav_planner::clampedHistoryCutoffSec;
using apexnav_planner::isAcceptableMapSourceStamp;
using apexnav_planner::isFinitePoseAndReasonableQuaternion;
using apexnav_planner::isFiniteXYZ32PointCloud;
using apexnav_planner::isInsideSelfFootprint;
using apexnav_planner::isSourceStampCoherent;
using apexnav_planner::isValidNestedCloudContract;
using apexnav_planner::isValidMapFrameContract;
using apexnav_planner::isValidSemanticDetectionMetadata;
using apexnav_planner::mappingTransitionRequiresFullReset;
using apexnav_planner::mayPublishEsdfSnapshot;
using apexnav_planner::rebuildInflationRegion;
using apexnav_planner::resetEpochGridBuffers;
using apexnav_planner::semanticFovConfidence;
using apexnav_planner::tryFuseSemanticValue;

TEST(MapFilterPolicy, FiltersOnlyYawAlignedFootprintAtYawZero)
{
  const SelfFilterFootprint footprint;
  EXPECT_TRUE(isInsideSelfFootprint(0.349999, 0.0, 0.0, 0.0, 0.0, footprint));
  EXPECT_TRUE(isInsideSelfFootprint(0.0, -0.35, 0.0, 0.0, 0.0, footprint));
  // These were incorrectly erased by the prior 0.65 m circular filter.
  EXPECT_FALSE(isInsideSelfFootprint(0.50, 0.40, 0.0, 0.0, 0.0, footprint));
  EXPECT_FALSE(isInsideSelfFootprint(0.64, 0.0, 0.0, 0.0, 0.0, footprint));
}

TEST(MapFilterPolicy, FiltersOnlyYawAlignedFootprintAfterRotation)
{
  const SelfFilterFootprint footprint;
  const double yaw = M_PI_2;
  EXPECT_TRUE(isInsideSelfFootprint(0.0, 0.349999, 0.0, 0.0, yaw, footprint));
  EXPECT_FALSE(isInsideSelfFootprint(0.40, 0.50, 0.0, 0.0, yaw, footprint));
  EXPECT_FALSE(isInsideSelfFootprint(0.0, 0.64, 0.0, 0.0, yaw, footprint));
}

TEST(MapFilterPolicy, RejectsZeroStaleFutureAndRegressingSourceStamps)
{
  EXPECT_TRUE(isAcceptableMapSourceStamp(100.0, 100.1, 99.0, 1.0, 0.05));
  EXPECT_FALSE(isAcceptableMapSourceStamp(0.0, 100.0, 0.0, 1.0, 0.05));
  EXPECT_FALSE(isAcceptableMapSourceStamp(98.9, 100.0, 0.0, 1.0, 0.05));
  EXPECT_FALSE(isAcceptableMapSourceStamp(100.1, 100.0, 0.0, 1.0, 0.05));
  EXPECT_FALSE(isAcceptableMapSourceStamp(99.0, 100.0, 99.0, 1.0, 0.05));
}

TEST(MapFilterPolicy, RebuildKeepsNeighbourHaloAfterClearingOverlappingObstacle)
{
  const int width = 9;
  const int height = 9;
  std::vector<unsigned char> occupied(width * height, 0);
  std::vector<unsigned char> inflated(width * height, 1);
  // A at (3,4) remains occupied; B at (5,4) was cleared. Rebuilding B's
  // former halo must retain all target cells still covered by A.
  occupied[3 * height + 4] = 1;
  rebuildInflationRegion(occupied, width, height, 3, 7, 2, 6, 2, &inflated);
  EXPECT_EQ(1, inflated[4 * height + 4]);  // overlap cell, covered by A
  EXPECT_EQ(1, inflated[3 * height + 6]);  // edge of A's halo
  EXPECT_EQ(0, inflated[7 * height + 4]);  // only B would have covered this
}

TEST(MapFilterPolicy, RaycastEpochIsWiderThanEightBits)
{
  EXPECT_GT(sizeof(RaycastEpoch), 1u);
}

TEST(MapFilterPolicy, RejectsNonFiniteAndBadNormPosesBeforeMapWrites)
{
  EXPECT_TRUE(isFinitePoseAndReasonableQuaternion(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.99));
  EXPECT_FALSE(isFinitePoseAndReasonableQuaternion(
      std::numeric_limits<double>::quiet_NaN(), 2.0, 3.0, 0.0, 0.0, 0.0, 1.0));
  EXPECT_FALSE(isFinitePoseAndReasonableQuaternion(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0));
  EXPECT_FALSE(isFinitePoseAndReasonableQuaternion(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.10));
}

TEST(MapFilterPolicy, RequiresCoherentMapCameraAndDepthFrames)
{
  EXPECT_TRUE(isValidMapFrameContract(
      "map", "map", "iris_camera_optical_frame", "iris_camera_optical_frame",
      "iris_camera_optical_frame"));
  EXPECT_FALSE(isValidMapFrameContract(
      "map", "odom", "iris_camera_optical_frame", "iris_camera_optical_frame",
      "iris_camera_optical_frame"));
  EXPECT_FALSE(isValidMapFrameContract(
      "map", "map", "base_link", "iris_camera_optical_frame", "iris_camera_optical_frame"));
  EXPECT_FALSE(isValidMapFrameContract(
      "map", "map", "iris_camera_optical_frame", "", "iris_camera_optical_frame"));
  EXPECT_FALSE(isValidMapFrameContract(
      "map", "map", "iris_camera_optical_frame", "iris_camera_optical_frame", "camera"));
}

TEST(MapFilterPolicy, InflationUsesExactRadiusInsteadOfRoundedCellHalo)
{
  const int width = 5;
  const int height = 5;
  std::vector<unsigned char> occupied(width * height, 0);
  std::vector<unsigned char> inflated(width * height, 0);
  occupied[2 * height + 2] = 1;
  rebuildInflationRegion(occupied, width, height, 0, 4, 0, 4, 1.20, &inflated);
  EXPECT_EQ(1, inflated[3 * height + 2]);
  EXPECT_EQ(0, inflated[3 * height + 3]);  // sqrt(2) cells is outside 1.20 cells.
}

TEST(MapFilterPolicy, SemanticMetadataAndNestedCloudContractFailClosed)
{
  EXPECT_TRUE(isValidSemanticDetectionMetadata(0, 0.75));
  EXPECT_TRUE(isValidSemanticDetectionMetadata(4, 1.0));
  EXPECT_FALSE(isValidSemanticDetectionMetadata(-1, 0.75));
  EXPECT_FALSE(isValidSemanticDetectionMetadata(5, 0.75));
  EXPECT_FALSE(isValidSemanticDetectionMetadata(
      0, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(isValidSemanticDetectionMetadata(0, -0.01));
  EXPECT_FALSE(isValidSemanticDetectionMetadata(0, 1.01));

  EXPECT_TRUE(isValidNestedCloudContract(42.0, "map", 42.0, "map"));
  EXPECT_FALSE(isValidNestedCloudContract(42.0, "map", 0.0, "map"));
  EXPECT_FALSE(isValidNestedCloudContract(42.0, "map", 41.9, "map"));
  EXPECT_FALSE(isValidNestedCloudContract(42.0, "map", 42.0, "camera"));
}

TEST(MapFilterPolicy, SemanticCloudRequiresFiniteXYZ32AndConsistentDimensions)
{
  sensor_msgs::PointCloud2 cloud;
  cloud.width = 1;
  cloud.height = 1;
  cloud.point_step = 12;
  cloud.row_step = 12;
  cloud.fields.resize(3);
  const char* names[] = {"x", "y", "z"};
  for (size_t index = 0; index < 3; ++index) {
    cloud.fields[index].name = names[index];
    cloud.fields[index].offset = 4 * index;
    cloud.fields[index].datatype = sensor_msgs::PointField::FLOAT32;
    cloud.fields[index].count = 1;
  }
  cloud.data.resize(12);
  const float xyz[] = {1.0f, 2.0f, 3.0f};
  std::memcpy(cloud.data.data(), xyz, sizeof(xyz));
  EXPECT_TRUE(isFiniteXYZ32PointCloud(cloud));

  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(cloud.data.data() + 8, &nan, sizeof(nan));
  EXPECT_FALSE(isFiniteXYZ32PointCloud(cloud));
  std::memcpy(cloud.data.data() + 8, &xyz[2], sizeof(xyz[2]));
  cloud.fields[2].datatype = sensor_msgs::PointField::UINT32;
  EXPECT_FALSE(isFiniteXYZ32PointCloud(cloud));
  cloud.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  cloud.row_step = 8;
  EXPECT_FALSE(isFiniteXYZ32PointCloud(cloud));

  cloud.width = 0;
  cloud.row_step = 0;
  cloud.data.clear();
  EXPECT_TRUE(isFiniteXYZ32PointCloud(cloud));
}

TEST(MapFilterPolicy, CameraInfoAndDepthSourcesMustBeCoherent)
{
  EXPECT_TRUE(isSourceStampCoherent(20.0, 20.005, 0.01));
  EXPECT_FALSE(isSourceStampCoherent(0.0, 20.0, 0.01));
  EXPECT_FALSE(isSourceStampCoherent(20.0, 20.02, 0.01));
  EXPECT_FALSE(isSourceStampCoherent(
      std::numeric_limits<double>::infinity(), 20.0, 0.01));
}

TEST(MapFilterPolicy, EsdfSnapshotIsSuppressedWhileAnUpdateIsPending)
{
  EXPECT_TRUE(mayPublishEsdfSnapshot(false, false));
  EXPECT_FALSE(mayPublishEsdfSnapshot(true, false));
  EXPECT_FALSE(mayPublishEsdfSnapshot(false, true));
  EXPECT_FALSE(mayPublishEsdfSnapshot(true, true));
}

TEST(MapFilterPolicy, HistoryCutoffNeverUnderflowsRosTimeAtEarlySimulationEpoch)
{
  EXPECT_DOUBLE_EQ(0.0, clampedHistoryCutoffSec(2.0, 5.0));
  EXPECT_DOUBLE_EQ(0.0, clampedHistoryCutoffSec(5.0, 5.0));
  EXPECT_DOUBLE_EQ(2.0, clampedHistoryCutoffSec(7.0, 5.0));
  EXPECT_DOUBLE_EQ(0.0, clampedHistoryCutoffSec(7.0, -1.0));
}

TEST(MapFilterPolicy, NewMissionCannotRevivePreviousMissionMap)
{
  EXPECT_TRUE(mappingTransitionRequiresFullReset(true));
  EXPECT_FALSE(mappingTransitionRequiresFullReset(false));
}

TEST(MapFilterPolicy, EpochResetRestoresUnknownAndClearsDerivedGridState)
{
  constexpr size_t size = 3;
  std::vector<double> occupancy(size, 4.0);
  std::vector<char> inflated(size, 1);
  std::vector<double> distance_negative(size, 2.0);
  std::vector<double> distance(size, 3.0);
  std::vector<double> temporary(size, 5.0);
  std::vector<char> virtual_ground(size, 1);
  std::vector<short> hit(size, 7);
  std::vector<short> miss(size, 8);
  std::vector<short> hit_and_miss(size, 9);
  std::vector<RaycastEpoch> ray_flags(size, 10);

  ASSERT_TRUE(resetEpochGridBuffers(size, -7.5, 6.0, &occupancy, &inflated,
      &distance_negative, &distance, &temporary, &virtual_ground, &hit, &miss,
      &hit_and_miss, &ray_flags));
  EXPECT_EQ(std::vector<double>(size, -7.5), occupancy);
  EXPECT_EQ(std::vector<char>(size, 0), inflated);
  EXPECT_EQ(std::vector<double>(size, 6.0), distance_negative);
  EXPECT_EQ(std::vector<double>(size, 6.0), distance);
  EXPECT_EQ(std::vector<double>(size, 0.0), temporary);
  EXPECT_EQ(std::vector<char>(size, 0), virtual_ground);
  EXPECT_EQ(std::vector<short>(size, 0), hit);
  EXPECT_EQ(std::vector<short>(size, 0), miss);
  EXPECT_EQ(std::vector<short>(size, 0), hit_and_miss);
  EXPECT_EQ(std::vector<RaycastEpoch>(size, 0), ray_flags);
}

TEST(MapFilterPolicy, SemanticFovRejectsPointsOutsideCameraCone)
{
  const double fov = 79.0 * M_PI / 180.0;
  EXPECT_DOUBLE_EQ(1.0, semanticFovConfidence(0.0, fov));
  EXPECT_DOUBLE_EQ(0.0, semanticFovConfidence(M_PI, fov));
  EXPECT_DOUBLE_EQ(0.0, semanticFovConfidence(0.5 * fov + 1e-6, fov));
}

TEST(MapFilterPolicy, ZeroOrInvalidSemanticWeightCannotWriteNan)
{
  double confidence = 0.4;
  double value = 0.3;
  EXPECT_FALSE(tryFuseSemanticValue(0.0, 0.8, 0.0, 0.2, &confidence, &value));
  EXPECT_DOUBLE_EQ(0.4, confidence);
  EXPECT_DOUBLE_EQ(0.3, value);
  EXPECT_FALSE(tryFuseSemanticValue(std::numeric_limits<double>::quiet_NaN(),
      0.8, 0.2, 0.2, &confidence, &value));
  ASSERT_TRUE(tryFuseSemanticValue(1.0, 0.8, 0.5, 0.2, &confidence, &value));
  EXPECT_TRUE(std::isfinite(confidence));
  EXPECT_TRUE(std::isfinite(value));
  EXPECT_NEAR((1.0 * 0.8 + 0.5 * 0.2) / 1.5, value, 1e-12);
}

}  // namespace
