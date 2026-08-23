#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <exploration_manager/dynamic_voronoi_allocator.h>

using apexnav_planner::DynamicVoronoiAllocator;
using apexnav_planner::agentAtspStem;
using apexnav_planner::agentIdFromAtspProblemCode;
using apexnav_planner::atspProblemCode;
using apexnav_planner::VoronoiAgent;
using apexnav_planner::VoronoiConfig;
using apexnav_planner::VoronoiDisplayColor;
using apexnav_planner::VoronoiFrontier;
using apexnav_planner::VoronoiGrid;
using apexnav_planner::frontierAgentActive;
using apexnav_planner::partitionFrontierCandidates;
using apexnav_planner::voronoiDisplayColor;

namespace {

void expect(bool condition, const std::string& message)
{
  if (!condition)
    throw std::runtime_error(message);
}

VoronoiGrid openGrid(int width, int height)
{
  VoronoiGrid grid;
  grid.width = width;
  grid.height = height;
  grid.resolution = 1.0;
  grid.traversable.assign(width * height, 1);
  return grid;
}

int address(const VoronoiGrid& grid, int x, int y)
{
  return x * grid.height + y;
}

void testOpenMapHasUniqueSpatialOwners()
{
  DynamicVoronoiAllocator allocator{VoronoiConfig()};
  const VoronoiGrid grid = openGrid(7, 3);
  const std::vector<VoronoiAgent> agents = {{0, 0, 1, true}, {1, 6, 1, true}};
  const std::vector<VoronoiFrontier> frontiers = {{10, 1, 1, 0.2}, {11, 5, 1, 0.8}};

  const auto result = allocator.allocate(grid, agents, frontiers);

  expect(result.owner_by_cell[address(grid, 1, 1)] == 0, "left cell owner");
  expect(result.owner_by_cell[address(grid, 5, 1)] == 1, "right cell owner");
  expect(result.frontier_owner.size() == 2, "frontier owner count");
  expect(result.frontier_owner[0] == 0, "left frontier owner");
  expect(result.frontier_owner[1] == 1, "right frontier owner");
}

void testWallsUseGeodesicInsteadOfEuclideanDistance()
{
  VoronoiGrid grid = openGrid(7, 5);
  for (int y = 0; y < 4; ++y)
    grid.traversable[address(grid, 3, y)] = 0;
  const std::vector<VoronoiAgent> agents = {{0, 0, 0, true}, {1, 4, 1, true}};
  const std::vector<VoronoiFrontier> frontiers = {{20, 2, 1, 0.5}};

  DynamicVoronoiAllocator allocator{VoronoiConfig()};
  const auto result = allocator.allocate(grid, agents, frontiers);

  expect(result.frontier_owner[0] == 0, "wall must use geodesic owner");
}

void testDiagonalPropagationCannotCutBlockedCorners()
{
  VoronoiGrid grid = openGrid(2, 2);
  grid.traversable[address(grid, 0, 1)] = 0;
  grid.traversable[address(grid, 1, 0)] = 0;
  const std::vector<VoronoiAgent> agents = {{0, 0, 0, true}};
  const std::vector<VoronoiFrontier> frontiers = {{21, 1, 1, 0.5}};

  DynamicVoronoiAllocator allocator{VoronoiConfig()};
  const auto result = allocator.allocate(grid, agents, frontiers);

  expect(result.frontier_owner[0] == -1, "blocked diagonal must stay unreachable");
}

void testInactiveAgentReleasesAllReachableCells()
{
  const VoronoiGrid grid = openGrid(5, 1);
  const std::vector<VoronoiAgent> agents = {{0, 0, 0, true}, {1, 4, 0, false}};
  const std::vector<VoronoiFrontier> frontiers = {{30, 4, 0, 0.5}};

  DynamicVoronoiAllocator allocator{VoronoiConfig()};
  const auto result = allocator.allocate(grid, agents, frontiers);

  expect(result.frontier_owner[0] == 0, "inactive agent frontier released");
  expect(result.owner_by_cell[address(grid, 4, 0)] == 0, "inactive agent cells released");
}

void testSemanticWorkloadProducesBoundedBoundaryBias()
{
  VoronoiConfig config;
  config.max_bias_m = 3.0;
  config.semantic_weight = 1.0;
  DynamicVoronoiAllocator allocator(config);
  const VoronoiGrid grid = openGrid(11, 1);
  const std::vector<VoronoiAgent> agents = {{0, 0, 0, true}, {1, 10, 0, true}};
  const std::vector<VoronoiFrontier> frontiers = {
      {40, 1, 0, 1.0}, {41, 4, 0, 0.9}, {42, 9, 0, 0.0}};

  const auto result = allocator.allocate(grid, agents, frontiers);

  expect(result.biases.size() == 2, "bias count");
  expect(result.biases[0] > result.biases[1], "semantic workload bias direction");
  expect(std::fabs(result.biases[0]) <= 3.0 + 1e-9, "agent 0 bias bound");
  expect(std::fabs(result.biases[1]) <= 3.0 + 1e-9, "agent 1 bias bound");
  expect(result.loads[0] > 0.0, "agent 0 nonzero load");
  expect(result.loads[1] > 0.0, "agent 1 nonzero load");
}

void testOwnershipHysteresisKeepsSmallBoundaryChange()
{
  VoronoiConfig config;
  config.max_balance_iterations = 0;
  config.owner_hysteresis_m = 0.3;
  DynamicVoronoiAllocator allocator(config);
  VoronoiGrid grid = openGrid(41, 1);
  grid.resolution = 0.1;
  const std::vector<VoronoiFrontier> no_frontiers;

  allocator.allocate(grid, {{0, 0, 0, true}, {1, 40, 0, true}}, no_frontiers);
  const auto result = allocator.allocate(
      grid, {{0, 0, 0, true}, {1, 39, 0, true}}, no_frontiers);

  expect(result.owner_by_cell[address(grid, 20, 0)] == 0, "ownership hysteresis");
}

void testAgentAtspFilesAreIndependent()
{
  expect(agentAtspStem(0) == "atsp_tour", "agent 0 keeps legacy ATSP stem");
  expect(agentAtspStem(1) == "atsp_tour_agent_1", "agent 1 ATSP stem");
  expect(atspProblemCode(0) == 1, "legacy ATSP request code remains agent 0");
  expect(atspProblemCode(1) == 2, "agent 1 ATSP request code");
  expect(agentIdFromAtspProblemCode(1) == 0, "decode legacy ATSP code");
  expect(agentIdFromAtspProblemCode(2) == 1, "decode agent 1 ATSP code");
}

void testHysteresisDoesNotCrossAChangedWorldGrid()
{
  VoronoiConfig config;
  config.max_balance_iterations = 0;
  config.owner_hysteresis_m = 0.3;
  DynamicVoronoiAllocator allocator(config);
  VoronoiGrid grid = openGrid(41, 1);
  grid.resolution = 0.1;
  allocator.allocate(grid, {{0, 0, 0, true}, {1, 40, 0, true}}, {});

  grid.offset_x = 100;
  const auto result = allocator.allocate(
      grid, {{0, 0, 0, true}, {1, 39, 0, true}}, {});
  expect(result.owner_by_cell[address(grid, 20, 0)] == 1,
      "hysteresis cache must reset when cropped world-grid geometry changes");
}

void testObjectSearchTemporarilyRemovesVoronoiSeed()
{
  expect(frontierAgentActive(true, false, false, false), "normal frontier agent active");
  expect(!frontierAgentActive(true, false, true, false), "object search seed removed");
  expect(!frontierAgentActive(true, true, false, false), "finished seed removed");
  expect(!frontierAgentActive(true, false, false, true), "terminal seed removed");
}

void testFrontierPartitionKeepsUniqueOwnershipAndUnlockedFallbacks()
{
  const auto partition = partitionFrontierCandidates(
      {0, 1, -1, 0}, {-1, -1, -1, 1}, 0);
  expect(partition.owned_indices == std::vector<int>{0}, "owned candidate partition");
  expect(partition.fallback_indices == std::vector<int>({1, 2}), "fallback candidate partition");
}

void testBlockedSeedProjectsToNearestTraversableCell()
{
  VoronoiGrid grid = openGrid(7, 1);
  grid.traversable[address(grid, 2, 0)] = 0;
  DynamicVoronoiAllocator allocator{VoronoiConfig()};

  const auto result = allocator.allocate(grid, {{0, 2, 0, true}, {1, 6, 0, true}},
      {{50, 1, 0, 0.5}, {51, 6, 0, 0.5}});
  expect(result.owner_by_cell[address(grid, 1, 0)] == 0,
      "blocked seed must acquire its nearest traversable cell");
  expect(result.owner_by_cell[address(grid, 6, 0)] == 1,
      "the other robot must retain its own region");
  expect(result.frontier_owner == std::vector<int>({0, 1}),
      "projected seed must produce a coarse two-robot frontier partition");
}

void testAllocationReportsProjectedSeedDiagnostics()
{
  VoronoiGrid grid = openGrid(3, 1);
  grid.traversable[address(grid, 1, 0)] = 0;
  DynamicVoronoiAllocator allocator{VoronoiConfig()};

  const auto result = allocator.allocate(grid, {{0, 1, 0, true}}, {});

  expect(result.seed_addresses == std::vector<int>({0}),
      "diagnostics must record the effective traversable seed address");
  expect(result.seed_projected == std::vector<bool>({true}),
      "diagnostics must record when a blocked seed was projected");
}

void testExclusiveActiveFrontiersGiveBothActiveAgentsDistinctTargets()
{
  const VoronoiGrid grid = openGrid(9, 1);
  DynamicVoronoiAllocator allocator{VoronoiConfig()};

  const auto result = allocator.allocate(grid, {{0, 0, 0, true}, {1, 0, 0, true}},
      {{54, 2, 0, 0.5, true}, {55, 7, 0, 0.5, true}});

  expect(result.frontier_owner.size() == 2, "active frontier owner count");
  expect(result.frontier_owner[0] != result.frontier_owner[1],
      "active agents must receive distinct frontier targets when two exist");
  expect(result.enforced_frontier_reassignments == 1,
      "exclusive assignment must report the reassigned active frontier");
}

void testBlockedFrontierRemainsInvalidWithoutClusterProjection()
{
  VoronoiGrid grid = openGrid(5, 1);
  grid.traversable[address(grid, 2, 0)] = 0;
  DynamicVoronoiAllocator allocator{VoronoiConfig()};

  const auto invalid_frontier = allocator.allocate(
      grid, {{7, 0, 0, true}}, {{51, 2, 0, 0.5}});
  expect(invalid_frontier.frontier_owner[0] == -1,
      "blocked frontier must be explicitly projected by its real cluster");
}

void testFrontierRegionMaskLeavesIrrelevantKnownSpaceUnassigned()
{
  VoronoiGrid grid = openGrid(9, 1);
  for (int x = 0; x < 3; ++x)
    grid.traversable[address(grid, x, 0)] = 0;
  for (int x = 6; x < 9; ++x)
    grid.traversable[address(grid, x, 0)] = 0;

  DynamicVoronoiAllocator allocator{VoronoiConfig()};
  const auto result = allocator.allocate(grid, {{0, 0, 0, true}, {1, 8, 0, true}},
      {{52, 3, 0, 0.5}, {53, 5, 0, 0.5}});

  expect(result.owner_by_cell[address(grid, 0, 0)] == -1,
      "known space outside the active frontier region must stay unassigned");
  expect(result.owner_by_cell[address(grid, 3, 0)] >= 0,
      "active frontier region must remain assignable");
  expect(result.owner_by_cell[address(grid, 5, 0)] >= 0,
      "active frontier region must remain assignable");
}

void testNonContiguousAgentIdsRemainValidOwners()
{
  DynamicVoronoiAllocator allocator{VoronoiConfig()};
  const VoronoiGrid grid = openGrid(5, 1);
  const auto result = allocator.allocate(
      grid, {{7, 0, 0, true}, {42, 4, 0, true}}, {{60, 1, 0, 0.5}, {61, 3, 0, 0.5}});
  expect(result.frontier_owner == std::vector<int>({7, 42}),
      "public owners must use stable agent ids, not vector indices");
}

void testVoronoiDisplayColorsRemainDistinctFromFreeSpaceBlue()
{
  const VoronoiDisplayColor agent_zero = voronoiDisplayColor(0);
  const VoronoiDisplayColor agent_one = voronoiDisplayColor(1);
  expect(agent_zero.r == 0 && agent_zero.g == 229 && agent_zero.b == 255,
      "agent 0 responsibility region must use vivid cyan");
  expect(agent_one.r == 255 && agent_one.g == 69 && agent_one.b == 0,
      "agent 1 responsibility region must use vivid orange-red");
}

}  // namespace

int main()
{
  testOpenMapHasUniqueSpatialOwners();
  testWallsUseGeodesicInsteadOfEuclideanDistance();
  testDiagonalPropagationCannotCutBlockedCorners();
  testInactiveAgentReleasesAllReachableCells();
  testSemanticWorkloadProducesBoundedBoundaryBias();
  testOwnershipHysteresisKeepsSmallBoundaryChange();
  testAgentAtspFilesAreIndependent();
  testHysteresisDoesNotCrossAChangedWorldGrid();
  testObjectSearchTemporarilyRemovesVoronoiSeed();
  testFrontierPartitionKeepsUniqueOwnershipAndUnlockedFallbacks();
  testBlockedSeedProjectsToNearestTraversableCell();
  testAllocationReportsProjectedSeedDiagnostics();
  testExclusiveActiveFrontiersGiveBothActiveAgentsDistinctTargets();
  testBlockedFrontierRemainsInvalidWithoutClusterProjection();
  testFrontierRegionMaskLeavesIrrelevantKnownSpaceUnassigned();
  testNonContiguousAgentIdsRemainValidOwners();
  testVoronoiDisplayColorsRemainDistinctFromFreeSpaceBlue();
  std::cout << "dynamic_voronoi_allocator_test: OK\n";
  return 0;
}
