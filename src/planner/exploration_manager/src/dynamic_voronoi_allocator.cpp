#include <exploration_manager/dynamic_voronoi_allocator.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace apexnav_planner {
namespace {

const double kInfinity = std::numeric_limits<double>::infinity();

double clampValue(double value, double lower, double upper)
{
  return std::max(lower, std::min(upper, value));
}

}  // namespace

std::string agentAtspStem(int agent_id)
{
  return agent_id == 0 ? "atsp_tour" : "atsp_tour_agent_" + std::to_string(agent_id);
}

int atspProblemCode(int agent_id)
{
  return agent_id + 1;
}

int agentIdFromAtspProblemCode(int problem_code)
{
  return problem_code - 1;
}

VoronoiDisplayColor voronoiDisplayColor(int agent_id)
{
  return agent_id == 0 ? VoronoiDisplayColor{0, 229, 255}
                       : VoronoiDisplayColor{255, 69, 0};
}

bool frontierAgentActive(
    bool have_odometry, bool finished, bool searching_object, bool terminal_state)
{
  return have_odometry && !finished && !searching_object && !terminal_state;
}

FrontierPartition partitionFrontierCandidates(const std::vector<int>& voronoi_owner,
    const std::vector<int>& claimed_by, int agent_id)
{
  if (voronoi_owner.size() != claimed_by.size())
    throw std::invalid_argument("Frontier owner and claim arrays must have equal length");
  FrontierPartition partition;
  for (int i = 0; i < static_cast<int>(voronoi_owner.size()); ++i) {
    if (claimed_by[i] >= 0 && claimed_by[i] != agent_id)
      continue;
    if (voronoi_owner[i] == agent_id)
      partition.owned_indices.push_back(i);
    else
      partition.fallback_indices.push_back(i);
  }
  return partition;
}

DynamicVoronoiAllocator::DynamicVoronoiAllocator(const VoronoiConfig& config) : config_(config)
{
}

void DynamicVoronoiAllocator::reset()
{
  previous_owner_by_cell_.clear();
  previous_width_ = 0;
  previous_height_ = 0;
  previous_offset_x_ = 0;
  previous_offset_y_ = 0;
}

int DynamicVoronoiAllocator::address(const VoronoiGrid& grid, int x, int y) const
{
  return x * grid.height + y;
}

int DynamicVoronoiAllocator::nearestTraversableAddress(
    const VoronoiGrid& grid, int x, int y) const
{
  if (x < 0 || x >= grid.width || y < 0 || y >= grid.height)
    return -1;
  const int adr = address(grid, x, y);
  return grid.traversable[adr] ? adr : -1;
}

int DynamicVoronoiAllocator::nearestTraversableSeedAddress(
    const VoronoiGrid& grid, int x, int y) const
{
  const int clamped_x = std::max(0, std::min(x, grid.width - 1));
  const int clamped_y = std::max(0, std::min(y, grid.height - 1));
  int nearest_address = -1;
  int nearest_squared_distance = std::numeric_limits<int>::max();
  for (int candidate_x = 0; candidate_x < grid.width; ++candidate_x) {
    for (int candidate_y = 0; candidate_y < grid.height; ++candidate_y) {
      const int candidate_address = address(grid, candidate_x, candidate_y);
      if (!grid.traversable[candidate_address])
        continue;
      const int dx = candidate_x - clamped_x;
      const int dy = candidate_y - clamped_y;
      const int squared_distance = dx * dx + dy * dy;
      if (squared_distance < nearest_squared_distance ||
          (squared_distance == nearest_squared_distance && candidate_address < nearest_address)) {
        nearest_squared_distance = squared_distance;
        nearest_address = candidate_address;
      }
    }
  }
  return nearest_address;
}

std::vector<double> DynamicVoronoiAllocator::computeDistanceField(
    const VoronoiGrid& grid, const VoronoiAgent& agent, int seed_address) const
{
  const int cell_count = grid.width * grid.height;
  std::vector<double> distance(cell_count, kInfinity);
  if (seed_address < 0)
    return distance;

  typedef std::pair<double, int> QueueEntry;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
  distance[seed_address] = 0.0;
  queue.emplace(0.0, seed_address);

  static const int offsets[8][2] = {
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  while (!queue.empty()) {
    const double current_distance = queue.top().first;
    const int current_address = queue.top().second;
    queue.pop();
    if (current_distance > distance[current_address] + 1e-9)
      continue;

    const int x = current_address / grid.height;
    const int y = current_address % grid.height;
    for (const auto& offset : offsets) {
      const int nx = x + offset[0];
      const int ny = y + offset[1];
      if (nx < 0 || nx >= grid.width || ny < 0 || ny >= grid.height)
        continue;
      const int next_address = address(grid, nx, ny);
      if (!grid.traversable[next_address])
        continue;
      const bool diagonal = offset[0] != 0 && offset[1] != 0;
      if (diagonal &&
          (!grid.traversable[address(grid, x + offset[0], y)] ||
              !grid.traversable[address(grid, x, y + offset[1])]))
        continue;
      const double edge_cost = grid.resolution * (diagonal ? std::sqrt(2.0) : 1.0);
      const double next_distance = current_distance + edge_cost;
      if (next_distance + 1e-9 < distance[next_address]) {
        distance[next_address] = next_distance;
        queue.emplace(next_distance, next_address);
      }
    }
  }
  return distance;
}

std::vector<int> DynamicVoronoiAllocator::computeOwners(const VoronoiGrid& grid,
    const std::vector<VoronoiAgent>& agents, const std::vector<std::vector<double>>& distances,
    const std::vector<double>& biases, bool apply_hysteresis) const
{
  const int cell_count = grid.width * grid.height;
  std::vector<int> owners(cell_count, -1);
  for (int adr = 0; adr < cell_count; ++adr) {
    if (!grid.traversable[adr])
      continue;

    int best_agent = -1;
    double best_cost = kInfinity;
    for (int i = 0; i < static_cast<int>(agents.size()); ++i) {
      if (!agents[i].active || !std::isfinite(distances[i][adr]))
        continue;
      const double cost = distances[i][adr] + biases[i];
      if (cost < best_cost - 1e-9 ||
          (std::fabs(cost - best_cost) <= 1e-9 &&
              (best_agent < 0 || agents[i].id < agents[best_agent].id))) {
        best_cost = cost;
        best_agent = i;
      }
    }

    if (apply_hysteresis && previous_owner_by_cell_.size() == owners.size()) {
      const int previous_agent_id = previous_owner_by_cell_[adr];
      int previous_agent = -1;
      for (int i = 0; i < static_cast<int>(agents.size()); ++i) {
        if (agents[i].id == previous_agent_id) {
          previous_agent = i;
          break;
        }
      }
      if (previous_agent >= 0 && agents[previous_agent].active &&
          std::isfinite(distances[previous_agent][adr])) {
        const double previous_cost = distances[previous_agent][adr] + biases[previous_agent];
        if (previous_cost <= best_cost + config_.owner_hysteresis_m)
          best_agent = previous_agent;
      }
    }
    owners[adr] = best_agent;
  }
  return owners;
}

std::vector<double> DynamicVoronoiAllocator::computeLoads(const VoronoiGrid& grid,
    const std::vector<VoronoiFrontier>& frontiers, const std::vector<int>& owners,
    const std::vector<std::vector<double>>& distances, int agent_count) const
{
  std::vector<double> loads(agent_count, 0.0);
  if (frontiers.empty())
    return loads;

  double semantic_min = kInfinity;
  double semantic_max = -kInfinity;
  for (const auto& frontier : frontiers) {
    semantic_min = std::min(semantic_min, frontier.semantic_value);
    semantic_max = std::max(semantic_max, frontier.semantic_value);
  }

  for (const auto& frontier : frontiers) {
    const int frontier_address = nearestTraversableAddress(grid, frontier.x, frontier.y);
    if (frontier_address < 0)
      continue;
    const int owner_index = owners[frontier_address];
    if (owner_index < 0 || owner_index >= agent_count ||
        !std::isfinite(distances[owner_index][frontier_address]))
      continue;
    const double semantic_norm = semantic_max - semantic_min > 1e-9
                                     ? (frontier.semantic_value - semantic_min) /
                                           (semantic_max - semantic_min)
                                     : 0.5;
    const double distance =
        std::min(distances[owner_index][frontier_address], config_.distance_cap_m);
    const double work = (1.0 + config_.semantic_weight * semantic_norm) *
                        (1.0 + distance / std::max(config_.distance_scale_m, 1e-6));
    loads[owner_index] += work;
  }
  return loads;
}

int DynamicVoronoiAllocator::enforceExclusiveActiveFrontiers(const VoronoiGrid& grid,
    const std::vector<VoronoiAgent>& agents, const std::vector<std::vector<double>>& distances,
    const std::vector<double>& biases, const std::vector<VoronoiFrontier>& frontiers,
    std::vector<int>& frontier_owners) const
{
  int active_agent_count = 0;
  int active_frontier_count = 0;
  for (const auto& agent : agents)
    if (agent.active)
      ++active_agent_count;
  for (const auto& frontier : frontiers)
    if (frontier.active)
      ++active_frontier_count;
  if (active_agent_count <= 1 || active_frontier_count < active_agent_count ||
      frontier_owners.size() != frontiers.size())
    return 0;

  std::vector<int> owned_active_frontier_counts(agents.size(), 0);
  for (int frontier_index = 0; frontier_index < static_cast<int>(frontiers.size()); ++frontier_index) {
    if (!frontiers[frontier_index].active)
      continue;
    for (int agent_index = 0; agent_index < static_cast<int>(agents.size()); ++agent_index) {
      if (frontier_owners[frontier_index] == agents[agent_index].id) {
        ++owned_active_frontier_counts[agent_index];
        break;
      }
    }
  }

  int reassignments = 0;
  for (int receiver_index = 0; receiver_index < static_cast<int>(agents.size()); ++receiver_index) {
    if (!agents[receiver_index].active || owned_active_frontier_counts[receiver_index] > 0)
      continue;

    int best_frontier_index = -1;
    int best_donor_index = -1;
    double best_cost_increase = kInfinity;
    for (int frontier_index = 0; frontier_index < static_cast<int>(frontiers.size()); ++frontier_index) {
      if (!frontiers[frontier_index].active)
        continue;
      const int frontier_address =
          nearestTraversableAddress(grid, frontiers[frontier_index].x, frontiers[frontier_index].y);
      if (frontier_address < 0 || !std::isfinite(distances[receiver_index][frontier_address]))
        continue;
      for (int donor_index = 0; donor_index < static_cast<int>(agents.size()); ++donor_index) {
        if (!agents[donor_index].active || owned_active_frontier_counts[donor_index] <= 1 ||
            frontier_owners[frontier_index] != agents[donor_index].id)
          continue;
        const double receiver_cost =
            distances[receiver_index][frontier_address] + biases[receiver_index];
        const double donor_cost = distances[donor_index][frontier_address] + biases[donor_index];
        const double cost_increase = receiver_cost - donor_cost;
        if (cost_increase < best_cost_increase) {
          best_cost_increase = cost_increase;
          best_frontier_index = frontier_index;
          best_donor_index = donor_index;
        }
      }
    }
    if (best_frontier_index >= 0) {
      frontier_owners[best_frontier_index] = agents[receiver_index].id;
      --owned_active_frontier_counts[best_donor_index];
      ++owned_active_frontier_counts[receiver_index];
      ++reassignments;
    }
  }
  return reassignments;
}

VoronoiResult DynamicVoronoiAllocator::allocate(const VoronoiGrid& grid,
    const std::vector<VoronoiAgent>& agents, const std::vector<VoronoiFrontier>& frontiers)
{
  if (grid.width <= 0 || grid.height <= 0 || grid.resolution <= 0.0 ||
      grid.traversable.size() != static_cast<size_t>(grid.width * grid.height))
    throw std::invalid_argument("Invalid Voronoi grid");

  if (grid.width != previous_width_ || grid.height != previous_height_ ||
      grid.offset_x != previous_offset_x_ || grid.offset_y != previous_offset_y_) {
    previous_owner_by_cell_.clear();
  }

  VoronoiResult result;
  result.distances.reserve(agents.size());
  result.seed_addresses.reserve(agents.size());
  result.seed_projected.reserve(agents.size());
  for (const auto& agent : agents) {
    const int raw_seed_address = nearestTraversableAddress(grid, agent.x, agent.y);
    const int seed_address = agent.active
                                 ? nearestTraversableSeedAddress(grid, agent.x, agent.y)
                                 : -1;
    result.seed_addresses.push_back(seed_address);
    result.seed_projected.push_back(agent.active && seed_address >= 0 &&
        seed_address != raw_seed_address);
    result.distances.push_back(computeDistanceField(grid, agent, seed_address));
  }
  result.biases.assign(agents.size(), 0.0);

  for (int iteration = 0; iteration < config_.max_balance_iterations; ++iteration) {
    const auto owners = computeOwners(grid, agents, result.distances, result.biases, false);
    const auto loads = computeLoads(grid, frontiers, owners, result.distances, agents.size());
    double active_load_sum = 0.0;
    int active_count = 0;
    for (int i = 0; i < static_cast<int>(agents.size()); ++i) {
      if (agents[i].active) {
        active_load_sum += loads[i];
        ++active_count;
      }
    }
    if (active_count <= 1 || active_load_sum <= 1e-9)
      break;
    const double mean_load = active_load_sum / active_count;
    for (int i = 0; i < static_cast<int>(agents.size()); ++i) {
      if (!agents[i].active)
        continue;
      const double target = clampValue(
          config_.balance_gain_m * (loads[i] - mean_load), -config_.max_bias_m, config_.max_bias_m);
      result.biases[i] = 0.5 * result.biases[i] + 0.5 * target;
    }
  }

  const auto owner_indices =
      computeOwners(grid, agents, result.distances, result.biases, true);
  result.loads = computeLoads(grid, frontiers, owner_indices, result.distances, agents.size());
  result.owner_by_cell.resize(owner_indices.size(), -1);
  for (int adr = 0; adr < static_cast<int>(owner_indices.size()); ++adr) {
    const int owner_index = owner_indices[adr];
    if (owner_index >= 0)
      result.owner_by_cell[adr] = agents[owner_index].id;
  }
  result.frontier_owner.reserve(frontiers.size());
  for (const auto& frontier : frontiers) {
    const int frontier_address = nearestTraversableAddress(grid, frontier.x, frontier.y);
    result.frontier_owner.push_back(frontier_address < 0
                                        ? -1
                                        : result.owner_by_cell[frontier_address]);
  }
  result.enforced_frontier_reassignments = enforceExclusiveActiveFrontiers(
      grid, agents, result.distances, result.biases, frontiers, result.frontier_owner);
  previous_owner_by_cell_ = result.owner_by_cell;
  previous_width_ = grid.width;
  previous_height_ = grid.height;
  previous_offset_x_ = grid.offset_x;
  previous_offset_y_ = grid.offset_y;
  return result;
}

}  // namespace apexnav_planner
