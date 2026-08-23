#ifndef _DYNAMIC_VORONOI_ALLOCATOR_H_
#define _DYNAMIC_VORONOI_ALLOCATOR_H_

#include <string>
#include <vector>

namespace apexnav_planner {

std::string agentAtspStem(int agent_id);
int atspProblemCode(int agent_id);
int agentIdFromAtspProblemCode(int problem_code);
struct VoronoiDisplayColor {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};
VoronoiDisplayColor voronoiDisplayColor(int agent_id);
bool frontierAgentActive(
    bool have_odometry, bool finished, bool searching_object, bool terminal_state);

struct FrontierPartition {
  std::vector<int> owned_indices;
  std::vector<int> fallback_indices;
};

FrontierPartition partitionFrontierCandidates(const std::vector<int>& voronoi_owner,
    const std::vector<int>& claimed_by, int agent_id);

struct VoronoiConfig {
  double max_bias_m = 3.0;
  double owner_hysteresis_m = 0.3;
  double semantic_weight = 1.0;
  double distance_scale_m = 10.0;
  double distance_cap_m = 20.0;
  double balance_gain_m = 1.0;
  int max_balance_iterations = 5;
};

struct VoronoiGrid {
  int width = 0;
  int height = 0;
  int offset_x = 0;
  int offset_y = 0;
  double resolution = 1.0;
  std::vector<unsigned char> traversable;
};

struct VoronoiAgent {
  int id = -1;
  int x = 0;
  int y = 0;
  bool active = false;
};

struct VoronoiFrontier {
  int id = -1;
  int x = 0;
  int y = 0;
  double semantic_value = 0.0;
  bool active = true;
};

struct VoronoiResult {
  std::vector<int> owner_by_cell;
  std::vector<int> frontier_owner;
  std::vector<double> biases;
  std::vector<double> loads;
  std::vector<std::vector<double>> distances;
  std::vector<int> seed_addresses;
  std::vector<bool> seed_projected;
  int enforced_frontier_reassignments = 0;
};

class DynamicVoronoiAllocator {
public:
  explicit DynamicVoronoiAllocator(const VoronoiConfig& config);

  VoronoiResult allocate(const VoronoiGrid& grid, const std::vector<VoronoiAgent>& agents,
      const std::vector<VoronoiFrontier>& frontiers);
  void reset();

private:
  int address(const VoronoiGrid& grid, int x, int y) const;
  int nearestTraversableAddress(const VoronoiGrid& grid, int x, int y) const;
  int nearestTraversableSeedAddress(const VoronoiGrid& grid, int x, int y) const;
  std::vector<double> computeDistanceField(
      const VoronoiGrid& grid, const VoronoiAgent& agent, int seed_address) const;
  std::vector<int> computeOwners(const VoronoiGrid& grid,
      const std::vector<VoronoiAgent>& agents, const std::vector<std::vector<double>>& distances,
      const std::vector<double>& biases, bool apply_hysteresis) const;
  std::vector<double> computeLoads(const VoronoiGrid& grid,
      const std::vector<VoronoiFrontier>& frontiers, const std::vector<int>& owners,
      const std::vector<std::vector<double>>& distances, int agent_count) const;
  int enforceExclusiveActiveFrontiers(const VoronoiGrid& grid,
      const std::vector<VoronoiAgent>& agents, const std::vector<std::vector<double>>& distances,
      const std::vector<double>& biases, const std::vector<VoronoiFrontier>& frontiers,
      std::vector<int>& frontier_owners) const;

  VoronoiConfig config_;
  std::vector<int> previous_owner_by_cell_;
  int previous_width_ = 0;
  int previous_height_ = 0;
  int previous_offset_x_ = 0;
  int previous_offset_y_ = 0;
};

}  // namespace apexnav_planner

#endif
