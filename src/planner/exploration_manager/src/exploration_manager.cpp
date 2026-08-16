/**
 * @file exploration_manager.cpp
 * @brief Implementation of exploration manager for autonomous semantic navigation
 * @author Zager-Zhang
 *
 * This file implements the ExplorationManager class that handles various
 * exploration strategies including distance-based, semantic-based, hybrid,
 * and TSP-optimized frontier selection for autonomous robot exploration.
 * 核心目标是根据环境感知信息（语义地图、前沿点、目标物体）选择最优探索目标，并规划可行路径 / 轨迹
 * (！！算法更改的核心区域！！)
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <lkh_mtsp_solver/SolveMTSP.h>
#include <plan_env/map_ros.h>
#include <path_searching/kino_astar.h>
#include <trajectory_manager/optimizer.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace Eigen;

namespace apexnav_planner {

ExplorationManager::~ExplorationManager() = default;

void ExplorationManager::initialize(ros::NodeHandle& nh)
{
  // Initialize SDF map and get object map reference
  sdf_map_.reset(new SDFMap2D);  // reset是智能指针的初始化方式
  sdf_map_->initMap(nh);
  object_map2d_ = sdf_map_->object_map2d_;  // 2D 物体语义地图,后续语义的基础

  // Initialize frontier map and path finder
  frontier_map2d_.reset(new FrontierMap2D(sdf_map_, nh));
  path_finder_.reset(new Astar2D);  //	A * 路径搜索器（2D）
  path_finder_->init(nh, sdf_map_);

  // Initialize exploration data and parameter containers
  ed_.reset(
      new ExplorationData);  // 探索数据容器，用于存储探索过程中的临时数据（如候选前沿点、路径、TSP
                             // 序列、语义目标列表）
  ep_.reset(new ExplorationParam);  // 探索参数容器，用于存储从 ROS 参数服务器读取的配置参数

  // Load exploration parameters from ROS parameter server
  nh.param("exploration/policy", ep_->policy_mode_, 0);  // 探索策略模式
  nh.param("exploration/sigma_threshold", ep_->sigma_threshold_,
      0.030);  // 语义值标准差阈值（混合策略判断）
  nh.param("exploration/max_to_mean_threshold", ep_->max_to_mean_threshold_, 1.2);
  nh.param("exploration/max_to_mean_percentage", ep_->max_to_mean_percentage_, 0.95);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));  // TSP 求解器的文件路径

  // Get map parameters for ray casting initialization（射线检测，进行碰撞校验）
  double resolution = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);

  // Initialize ray caster for collision checking and TSP service client
  ray_caster2d_.reset(new RayCaster2D);
  ray_caster2d_->setParams(resolution, origin);
  last_over_depth_object_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  tsp_client_ =
      nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp", true);  // TSP 求解服务客户端

  // Initialize KinoAstar and GCopter for real-world trajectory planning(动力学 A* +
  // 轨迹优化器，生成平滑可行的 3D 轨迹)
  kinoastar_.reset(new KinoAstar(nh, sdf_map_));
  kinoastar_->init();

  Config gcopter_config(nh);
  gcopter_.reset(new Gcopter(gcopter_config, nh, sdf_map_, kinoastar_));

  ROS_INFO("[ExplorationManager] KinoAstar and GCopter initialized for real-world mode");
}

void ExplorationManager::resetEpisodeState()
{
  last_over_depth_object_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

const char* ExplorationManager::navigationModeName(NavigationMode mode)
{
  switch (mode) {
    case NavigationMode::SEARCH_BEST_OBJECT: return "SEARCH_BEST_OBJECT";
    case NavigationMode::SEARCH_OVER_DEPTH_OBJECT: return "SEARCH_OVER_DEPTH_OBJECT";
    case NavigationMode::SEARCH_SUSPICIOUS_OBJECT: return "SEARCH_SUSPICIOUS_OBJECT";
    case NavigationMode::SEMANTIC_FRONTIER: return "SEMANTIC_FRONTIER";
    case NavigationMode::GEOMETRIC_FRONTIER: return "GEOMETRIC_FRONTIER";
    case NavigationMode::DORMANT_FRONTIER: return "DORMANT_FRONTIER";
    case NavigationMode::SEARCH_EXTREME: return "SEARCH_EXTREME";
    default: return "NONE";
  }
}

NavigationMode ExplorationManager::evaluateNavigationMode(const Vector3d& pos, int agent_idx)
{
  (void)agent_idx;
  Vector2d target;
  vector<Vector2d> path;
  const auto any_object_path = [&](const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& clouds,
                                   bool extreme) {
    for (const auto& cloud : clouds) {
      if (!cloud->points.empty() && (extreme
          ? searchObjectPathExtreme(pos, cloud, target, path)
          : searchObjectPath(pos, cloud, target, path))) return true;
    }
    return false;
  };

  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds;
  object_map2d_->getTopConfidenceObjectCloud(clouds);
  if (any_object_path(clouds, false)) return NavigationMode::SEARCH_BEST_OBJECT;
  if (!object_map2d_->over_depth_object_cloud_->points.empty() &&
      searchObjectPath(pos, object_map2d_->over_depth_object_cloud_, target, path))
    return NavigationMode::SEARCH_OVER_DEPTH_OBJECT;

  vector<Vector2d> reachable_frontiers;
  for (const Vector2d& frontier : ed_->frontier_averages_) {
    if (searchFrontierPath(Vector2d(pos.x(), pos.y()), frontier, target, path))
      reachable_frontiers.push_back(frontier);
  }
  if (!reachable_frontiers.empty()) {
    if (ep_->policy_mode_ == ExplorationParam::SEMANTIC)
      return NavigationMode::SEMANTIC_FRONTIER;
    if (ep_->policy_mode_ == ExplorationParam::HYBRID) {
      vector<SemanticFrontier> semantic_frontiers;
      getSortedSemanticFrontiers(Vector2d(pos.x(), pos.y()), reachable_frontiers, semantic_frontiers);
      double std_dev = 0.0, max_to_mean = 0.0, mean = 0.0;
      calcSemanticFrontierInfo(semantic_frontiers, std_dev, max_to_mean, mean);
      if (std_dev > ep_->sigma_threshold_ && max_to_mean > ep_->max_to_mean_threshold_)
        return NavigationMode::SEMANTIC_FRONTIER;
    }
    return NavigationMode::GEOMETRIC_FRONTIER;
  }

  object_map2d_->getTopConfidenceObjectCloud(clouds, false);
  if (any_object_path(clouds, false)) return NavigationMode::SEARCH_SUSPICIOUS_OBJECT;
  for (const Vector2d& frontier : ed_->dormant_frontier_averages_) {
    if (searchFrontierPath(Vector2d(pos.x(), pos.y()), frontier, target, path))
      return NavigationMode::DORMANT_FRONTIER;
  }
  object_map2d_->getTopConfidenceObjectCloud(clouds, false, true);
  if (any_object_path(clouds, true)) return NavigationMode::SEARCH_EXTREME;
  if (!last_over_depth_object_cloud_->points.empty() &&
      searchObjectPathExtreme(pos, last_over_depth_object_cloud_, target, path))
    return NavigationMode::SEARCH_EXTREME;
  return NavigationMode::NONE;
}

bool ExplorationManager::collectJointCandidates(const std::array<Vector3d, NUM_AGENTS>& agent_positions,
    NavigationMode mode, vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>>& candidates)
{
  candidates.clear();
  const auto add_object_candidates = [&](const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& clouds,
                                         bool extreme) {
    for (size_t i = 0; i < clouds.size(); ++i) {
      if (clouds[i]->points.empty()) continue;
      JointCandidate candidate;
      candidate.target_id = static_cast<int>(i);
      candidate.target_type = "OBJECT";
      candidate.semantic_score = static_cast<double>(clouds.size() - i);
      bool reachable_by_any = false;
      candidate.initial_costs.fill(std::numeric_limits<double>::infinity());
      for (int agent = 0; agent < NUM_AGENTS; ++agent) {
        const bool found = extreme
            ? searchObjectPathExtreme(agent_positions[agent], clouds[i], candidate.target_positions[agent], candidate.initial_paths[agent])
            : searchObjectPath(agent_positions[agent], clouds[i], candidate.target_positions[agent], candidate.initial_paths[agent]);
        if (!found) continue;
        reachable_by_any = true;
        candidate.initial_costs[agent] = 0.0;
        for (size_t p = 1; p < candidate.initial_paths[agent].size(); ++p)
          candidate.initial_costs[agent] += (candidate.initial_paths[agent][p] - candidate.initial_paths[agent][p - 1]).norm();
      }
      if (reachable_by_any) candidates.push_back(candidate);
    }
  };

  switch (mode) {
    case NavigationMode::SEARCH_BEST_OBJECT: {
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds;
      object_map2d_->getTopConfidenceObjectCloud(clouds);
      add_object_candidates(clouds, false);
      break;
    }
    case NavigationMode::SEARCH_SUSPICIOUS_OBJECT: {
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds;
      object_map2d_->getTopConfidenceObjectCloud(clouds, false);
      add_object_candidates(clouds, false);
      break;
    }
    case NavigationMode::SEARCH_OVER_DEPTH_OBJECT: {
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds;
      if (!object_map2d_->over_depth_object_cloud_->points.empty()) clouds.push_back(object_map2d_->over_depth_object_cloud_);
      add_object_candidates(clouds, false);
      break;
    }
    case NavigationMode::SEARCH_EXTREME: {
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds;
      object_map2d_->getTopConfidenceObjectCloud(clouds, false, true);
      add_object_candidates(clouds, true);
      break;
    }
    case NavigationMode::SEMANTIC_FRONTIER:
    case NavigationMode::GEOMETRIC_FRONTIER:
    case NavigationMode::DORMANT_FRONTIER: {
      const vector<Vector2d>& source = mode == NavigationMode::DORMANT_FRONTIER
          ? ed_->dormant_frontier_averages_ : ed_->frontier_averages_;
      for (size_t i = 0; i < source.size(); ++i) {
        JointCandidate candidate;
        candidate.target_id = findFrontierIdByPosition(source[i], mode == NavigationMode::DORMANT_FRONTIER);
        candidate.target_type = mode == NavigationMode::DORMANT_FRONTIER ? "DORMANT_FRONTIER" : "FRONTIER";
        candidate.semantic_score = getFrontierSemanticValue(source[i]);
        bool reachable_by_any = false;
        candidate.initial_costs.fill(std::numeric_limits<double>::infinity());
        for (int agent = 0; agent < NUM_AGENTS; ++agent) {
          if (!searchFrontierPath(Vector2d(agent_positions[agent].x(), agent_positions[agent].y()), source[i],
                  candidate.target_positions[agent], candidate.initial_paths[agent])) {
            continue;
          }
          reachable_by_any = true;
          candidate.initial_costs[agent] = 0.0;
          for (size_t p = 1; p < candidate.initial_paths[agent].size(); ++p)
            candidate.initial_costs[agent] += (candidate.initial_paths[agent][p] - candidate.initial_paths[agent][p - 1]).norm();
        }
        if (reachable_by_any) candidates.push_back(candidate);
      }
      break;
    }
    default: break;
  }

  // Semantic mode admits only the high-value portion used by the hybrid policy.
  if (mode == NavigationMode::SEMANTIC_FRONTIER && ep_->policy_mode_ == ExplorationParam::HYBRID && !candidates.empty()) {
    double max_score = 0.0, mean_score = 0.0;
    for (const auto& candidate : candidates) { max_score = std::max(max_score, candidate.semantic_score); mean_score += candidate.semantic_score; }
    mean_score /= candidates.size();
    const double threshold = std::max(ep_->max_to_mean_threshold_ * mean_score,
        ep_->max_to_mean_percentage_ * max_score);
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [&](const JointCandidate& candidate) { return candidate.semantic_score < threshold; }), candidates.end());
  }

  std::sort(candidates.begin(), candidates.end(), [mode](const JointCandidate& a, const JointCandidate& b) {
    if (mode == NavigationMode::SEMANTIC_FRONTIER) return a.semantic_score > b.semantic_score;
    return std::min(a.initial_costs[0], a.initial_costs[1]) < std::min(b.initial_costs[0], b.initial_costs[1]);
  });
  constexpr size_t kMaxJointCandidates = 10;
  if (candidates.size() > kMaxJointCandidates) candidates.resize(kMaxJointCandidates);
  return candidates.size() >= NUM_AGENTS;
}

bool ExplorationManager::computeJointMtspCostMatrix(
    const vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>>& candidates,
    std::array<Eigen::MatrixXd, NUM_AGENTS>& transition_costs)
{
  const int n = static_cast<int>(candidates.size());
  for (int agent = 0; agent < NUM_AGENTS; ++agent) {
    transition_costs[agent] = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      if (!std::isfinite(candidates[i].initial_costs[agent]) ||
          !std::isfinite(candidates[j].initial_costs[agent])) {
        transition_costs[agent](i, j) = std::numeric_limits<double>::infinity();
        continue;
      }
      const double cost = computePathCost(candidates[i].target_positions[agent], candidates[j].target_positions[agent]);
      // computePathCost uses 10000 as the legacy A* failure sentinel.  Do not
      // turn it into a finite mTSP penalty: an unreachable edge must never be
      // selected by the MINMAX dynamic program.
      transition_costs[agent](i, j) = (cost > 0.0 && cost < 10000.0)
          ? cost : std::numeric_limits<double>::infinity();
    }
  }
  return true;
}

bool ExplorationManager::solveTwoStartMinmax(
    const vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>>& candidates,
    const std::array<Eigen::MatrixXd, NUM_AGENTS>& transition_costs,
    std::array<vector<int>, NUM_AGENTS>& routes) const
{
  // held_karp DP: for each agent, DP[S][j] is the shortest open route from
  // its own start through subset S and ending at j.  Complementary subsets are
  // then selected by the MINMAX objective.
  const int n = static_cast<int>(candidates.size());
  if (n < NUM_AGENTS || n > 20) return false;
  const int states = 1 << n;
  // `inf` is a macro in the ROS/PCL include chain on some platforms.
  // Use a non-macro identifier so this planner remains portable.
  const double infinity_cost = std::numeric_limits<double>::infinity();
  std::array<vector<double>, NUM_AGENTS> best_subset;
  std::array<vector<int>, NUM_AGENTS> best_end;
  std::array<vector<double>, NUM_AGENTS> dp;
  std::array<vector<int>, NUM_AGENTS> parent;
  for (int agent = 0; agent < NUM_AGENTS; ++agent) {
    dp[agent].assign(states * n, infinity_cost); parent[agent].assign(states * n, -1);
    best_subset[agent].assign(states, infinity_cost); best_end[agent].assign(states, -1);
    for (int j = 0; j < n; ++j) {
      if (std::isfinite(candidates[j].initial_costs[agent]))
        dp[agent][(1 << j) * n + j] = candidates[j].initial_costs[agent];
    }
    for (int mask = 1; mask < states; ++mask) for (int last = 0; last < n; ++last) {
      const double current = dp[agent][mask * n + last];
      if (!std::isfinite(current)) continue;
      if (current < best_subset[agent][mask]) { best_subset[agent][mask] = current; best_end[agent][mask] = last; }
      for (int next = 0; next < n; ++next) if (!(mask & (1 << next))) {
        const double next_cost = current + transition_costs[agent](last, next);
        const int index = ((mask | (1 << next)) * n + next);
        if (next_cost < dp[agent][index]) { dp[agent][index] = next_cost; parent[agent][index] = last; }
      }
    }
  }
  const int full = states - 1; int selected = -1; double objective = infinity_cost;
  for (int mask = 1; mask < full; ++mask) {
    const double value = std::max(best_subset[0][mask], best_subset[1][full ^ mask]);
    if (value < objective) { objective = value; selected = mask; }
  }
  if (selected < 0 || !std::isfinite(objective)) return false;
  const std::array<int, NUM_AGENTS> subsets = {{ selected, full ^ selected }};
  for (int agent = 0; agent < NUM_AGENTS; ++agent) {
    int mask = subsets[agent], last = best_end[agent][mask];
    while (last >= 0) { routes[agent].push_back(last); const int previous = parent[agent][mask * n + last]; mask &= ~(1 << last); last = previous; }
    std::reverse(routes[agent].begin(), routes[agent].end());
  }
  return !routes[0].empty() && !routes[1].empty();
}

bool ExplorationManager::planJointModeTargets(const std::array<Vector3d, NUM_AGENTS>& agent_positions,
    NavigationMode mode, JointAssignment& assignment)
{
  assignment = JointAssignment();
  if (!agent_positions[0].allFinite() || !agent_positions[1].allFinite()) return false;
  if (mode == NavigationMode::NONE) return false;
  vector<JointCandidate, Eigen::aligned_allocator<JointCandidate>> candidates;
  if (!collectJointCandidates(agent_positions, mode, candidates)) return false;
  std::array<Eigen::MatrixXd, NUM_AGENTS> transition_costs;
  if (!computeJointMtspCostMatrix(candidates, transition_costs)) return false;
  std::array<vector<int>, NUM_AGENTS> routes;
  if (!solveTwoStartMinmax(candidates, transition_costs, routes)) return false;
  assignment.valid = true; assignment.mode = mode;
  for (int agent = 0; agent < NUM_AGENTS; ++agent) {
    const JointCandidate& first = candidates[routes[agent].front()];
    assignment.next_positions[agent] = first.target_positions[agent];
    assignment.next_paths[agent] = first.initial_paths[agent];
    setStrategyInfo(agent, "JOINT_MINMAX", first.target_type, first.target_id,
        first.semantic_score, first.initial_paths[agent], first.target_positions[agent]);
  }
  ROS_INFO("[Joint MINMAX] mode=%s candidates=%zu", navigationModeName(mode), candidates.size());
  return true;
}

int ExplorationManager::planNextBestPoint(const Vector3d& pos, const double& yaw, int agent_idx,
    Eigen::Vector2d& out_next_pos, std::vector<Eigen::Vector2d>& out_next_best_path)
{
  // 高置信度物体导航 → 过深物体导航 → 活跃前沿探索 → 可疑物体 → 休眠前沿 → 极端搜索 → 错误返回
  Vector2d pos2d = Vector2d(pos(0), pos(1));
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;

  // Clear previous planning results
  ed_->tsp_tour_.clear();
  out_next_best_path.clear();
  setStrategyInfo(agent_idx, "PLANNING", "NONE", -1, -1.0, out_next_best_path, pos2d);
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // ==================== Navigation Mode: High-Confidence Objects ====================

  if (!object_clouds.empty()) {  // 存在高置信度目标物体点云
    ROS_WARN("[Agent %d Navigation Mode] Get object_cloud num = %ld", agent_idx, object_clouds.size());

    // Try to find path to each detected object in order of confidence
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, out_next_pos, out_next_best_path)) {
        setStrategyInfo(agent_idx, "SEARCH_BEST_OBJECT", "OBJECT", -1, -1.0,
            out_next_best_path, out_next_pos);
        return SEARCH_BEST_OBJECT;
      }
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    ROS_WARN("[Agent %d Navigation Mode (Over Depth)] Get over depth object cloud", agent_idx);
    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, out_next_pos, out_next_best_path)) {
      setStrategyInfo(agent_idx, "SEARCH_OVER_DEPTH_OBJECT", "OBJECT", -1, -1.0,
          out_next_best_path, out_next_pos);
      return SEARCH_OVER_DEPTH_OBJECT;
    }
  }

  // ==================== Exploration Mode: Frontier-Based Planning ====================
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(
      object_clouds, false);
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (object_clouds.size() >= 1)
    top_object_cloud = object_clouds[0];

  // Apply selected exploration policy to choose next frontier
  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  chooseExplorationPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path, agent_idx);

  // Handle case when no passable frontiers are found
  if (next_best_path.empty()) {
    ROS_WARN("Agent %d: Maybe no passable frontier.", agent_idx);

    // Try suspicious objects as backup
    if (!top_object_cloud->points.empty() &&
        searchObjectPath(pos, top_object_cloud, out_next_pos, out_next_best_path)) {
      setStrategyInfo(agent_idx, "SEARCH_SUSPICIOUS_OBJECT", "OBJECT", -1, -1.0,
          out_next_best_path, out_next_pos);
      return SEARCH_SUSPICIOUS_OBJECT;
    }
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path, agent_idx);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      ROS_ERROR("Agent %d: search exterme case!!!", agent_idx);

      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, out_next_pos, out_next_best_path)) {
          setStrategyInfo(agent_idx, "SEARCH_EXTREME_OBJECT", "OBJECT", -1, -1.0,
              out_next_best_path, out_next_pos);
          return SEARCH_EXTREME;
        }
      }

      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, out_next_pos, out_next_best_path)) {
          setStrategyInfo(agent_idx, "SEARCH_EXTREME_OBJECT", "OBJECT", -1, -1.0,
              out_next_best_path, out_next_pos);
          return SEARCH_EXTREME;
        }
      }

      if (!object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud_ = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud_->points.empty() &&
          searchObjectPathExtreme(
              pos, last_over_depth_object_cloud_, out_next_pos, out_next_best_path)) {
        setStrategyInfo(agent_idx, "SEARCH_EXTREME_OVER_DEPTH_OBJECT", "OBJECT", -1, -1.0,
            out_next_best_path, out_next_pos);
        return SEARCH_EXTREME;
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        ROS_ERROR("Agent %d: No coverable frontier!!", agent_idx);
        setStrategyInfo(agent_idx, "NO_COVERABLE_FRONTIER", "NONE", -1, -1.0,
            out_next_best_path, pos2d);
        return NO_COVERABLE_FRONTIER;
      }
      else {
        ROS_ERROR("Agent %d: No passable frontier!!", agent_idx);
        setStrategyInfo(agent_idx, "NO_PASSABLE_FRONTIER", "NONE", -1, -1.0,
            out_next_best_path, pos2d);
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Store successful planning results
  out_next_pos = next_best_pos;
  out_next_best_path = next_best_path;

  // Performance monitoring
  double total_time = (ros::Time::now() - t2).toSec();
  ROS_ERROR_COND(total_time > 0.25, "[Agent %d Plan NBV] Total time %.2lf s too long!!!", agent_idx, total_time);

  return EXPLORATION;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx)
{
  // Different navigation modes are planned independently.  Deliberately do
  // not filter by a shared Claim here: equal modes are handled by the joint
  // MINMAX assignment in the FSM before this per-agent fallback is reached.

  switch (ep_->policy_mode_) {
    case ExplorationParam::DISTANCE:
      ROS_WARN("[Agent %d Exploration Mode] Find Closest Frontier", agent_idx);
      findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
      break;

    case ExplorationParam::SEMANTIC:
      ROS_WARN("[Agent %d Exploration Mode] Find Highest Semantic Value Frontier", agent_idx);
      findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
      break;

    case ExplorationParam::HYBRID:
      ROS_WARN("[Agent %d Exploration Mode] Working on Hybrid Mode", agent_idx);
      hybridExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
      break;

    case ExplorationParam::TSP_DIST:
      ROS_WARN("[Agent %d Exploration Mode] Working on TSP Distance Mode", agent_idx);
      findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
      break;

    default:
      ROS_WARN("[Agent %d Exploration Mode] Unknown Mode", agent_idx);
      break;
  }
}

void ExplorationManager::hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx)
{
  double std_dev_threshold = ep_->sigma_threshold_;
  double max_to_mean_threshold = ep_->max_to_mean_threshold_;
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  double std_dev, max_to_mean, mean;
  calcSemanticFrontierInfo(sem_frontiers, std_dev, max_to_mean, mean);

  // Decide between exploitation and exploration based on semantic statistics
  if (std_dev > std_dev_threshold && max_to_mean > max_to_mean_threshold) {
    ROS_WARN("Agent %d: Exploit the semantic value (TSP)!!", agent_idx);
    vector<Vector2d> high_sem_frontiers;

    // Select high-value frontiers for TSP optimization
    for (auto sem_frontier : sem_frontiers) {
      double auto_max_to_mean_threshold =
          max(max_to_mean_threshold, ep_->max_to_mean_percentage_ * max_to_mean);
      if (sem_frontier.semantic_value / mean < auto_max_to_mean_threshold)
        break;
      high_sem_frontiers.push_back(sem_frontier.position);
    }
    findTSPTourPolicy(cur_pos, high_sem_frontiers, next_best_pos, next_best_path, agent_idx);
    if (!next_best_path.empty()) {
      setStrategyInfo(agent_idx, "HYBRID_SEMANTIC_FRONTIER", "FRONTIER",
          findFrontierIdByPosition(next_best_pos), getFrontierSemanticValue(next_best_pos),
          next_best_path, next_best_pos);
    }
  }
  else {
    ROS_WARN("Agent %d: Explore the environment (Closest)!!", agent_idx);
    findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
    if (!next_best_path.empty()) {
      setStrategyInfo(agent_idx, "HYBRID_GEOMETRIC_FRONTIER", "FRONTIER",
          findFrontierIdByPosition(next_best_pos), getFrontierSemanticValue(next_best_pos),
          next_best_path, next_best_pos);
    }
  }
}

void ExplorationManager::findHighestSemanticsFrontierPolicy(Vector2d cur_pos,
    vector<Vector2d> frontiers, Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx)
{
  next_best_path.clear();

  // Container for frontier-value pairs for sorting
  vector<pair<Vector2d, double>> frontier_values;

  // Compute semantic value for each frontier
  for (auto frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 neighborhood

    // Find maximum semantic value in local neighborhood
    double value = sdf_map_->value_map_->getValue(idx);
    for (auto nbr : nbrs) value = max(value, sdf_map_->value_map_->getValue(nbr));

    frontier_values.emplace_back(frontier, value);
  }

  // Sort by semantic value (descending), then by distance (ascending)
  auto compareFrontiers = [&cur_pos](
                              const pair<Vector2d, double>& a, const pair<Vector2d, double>& b) {
    if (fabs(a.second - b.second) > 1e-5) {
      return a.second > b.second;  // Higher semantic value first
    }
    else {
      double dist_a = (a.first - cur_pos).norm();
      double dist_b = (b.first - cur_pos).norm();
      return dist_a < dist_b;  // Closer distance first for tie-breaking
    }
  };

  std::sort(frontier_values.begin(), frontier_values.end(), compareFrontiers);

  // Update frontier list with sorted order
  frontiers.clear();
  for (const auto& fv : frontier_values) {
    frontiers.push_back(fv.first);
  }

  // Select first reachable frontier from sorted list
  for (int i = 0; i < (int)frontiers.size(); i++) {
    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;
    next_best_pos = tmp_pos;
    next_best_path = tmp_path;
    setStrategyInfo(agent_idx, "SEMANTIC_FRONTIER", "FRONTIER",
        findFrontierIdByPosition(next_best_pos), getFrontierSemanticValue(next_best_pos),
        next_best_path, next_best_pos);
    break;
  }
}

void ExplorationManager::findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx)
{
  next_best_path.clear();

  // Sort frontiers by Euclidean distance for efficient processing
  std::sort(frontiers.begin(), frontiers.end(), [&cur_pos](const Vector2d& a, const Vector2d& b) {
    return (a - cur_pos).norm() < (b - cur_pos).norm();
  });

  double min_len = std::numeric_limits<double>::max();

  // Find the frontier with shortest actual path length
  for (int i = 0; i < (int)frontiers.size(); i++) {
    // Skip if Euclidean distance already exceeds best path length
    if ((frontiers[i] - cur_pos).norm() >= min_len)
      continue;

    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;

    // Attempt path planning to this frontier
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;

    // Update best solution if this path is shorter
    double len = Astar2D::pathLength(tmp_path);
    if (len < min_len) {
      min_len = len;
      next_best_pos = tmp_pos;
      next_best_path = tmp_path;
    }
  }
  if (!next_best_path.empty()) {
    setStrategyInfo(agent_idx, "GEOMETRIC_FRONTIER", "FRONTIER",
        findFrontierIdByPosition(next_best_pos), getFrontierSemanticValue(next_best_pos),
        next_best_path, next_best_pos);
  }
}

void ExplorationManager::findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx)
{
  next_best_path.clear();
  vector<Vector2d> filter_frontiers;
  for (auto frontier : frontiers) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path))
      filter_frontiers.push_back(frontier);
  }

  vector<int> indices;
  computeATSPTour(cur_pos, filter_frontiers, indices);
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : indices) ed_->tsp_tour_.push_back(filter_frontiers[idx]);

  if (!indices.empty()) {
    for (auto idx : indices) {
      Vector2d next_bext_frontier = filter_frontiers[idx];
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path)) {
        setStrategyInfo(agent_idx, "TSP_FRONTIER", "FRONTIER",
            findFrontierIdByPosition(next_best_pos), getFrontierSemanticValue(next_best_pos),
            next_best_path, next_best_pos);
        break;
      }
    }
  }
}

double ExplorationManager::getFrontierSemanticValue(const Vector2d& frontier)
{
  Vector2i idx;
  sdf_map_->posToIndex(frontier, idx);
  double value = sdf_map_->value_map_->getValue(idx);
  auto nbrs = allNeighbors(idx, 2);
  for (auto& nbr : nbrs) {
    if (sdf_map_->getInflateOccupancy(nbr) == 1 ||
        sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
      continue;
    value = std::max(value, sdf_map_->value_map_->getValue(nbr));
  }
  return value;
}

int ExplorationManager::findFrontierIdByPosition(const Vector2d& frontier, bool dormant)
{
  const auto& frontiers = dormant ? ed_->dormant_frontier_averages_ : ed_->frontier_averages_;
  if (frontiers.empty())
    return dormant ? -1 : findFrontierIdByPosition(frontier, true);

  int best_id = -1;
  double best_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(frontiers.size()); ++i) {
    double dist = (frontiers[i] - frontier).norm();
    if (dist < best_dist) {
      best_dist = dist;
      best_id = i;
    }
  }
  if (best_dist < 1e-2)
    return best_id;
  return dormant ? -1 : findFrontierIdByPosition(frontier, true);
}

void ExplorationManager::setStrategyInfo(int agent_idx, const std::string& mode,
    const std::string& target_type, int target_id, double semantic_score,
    const vector<Vector2d>& path, const Vector2d& target_pos)
{
  if (agent_idx < 0 || agent_idx >= static_cast<int>(ed_->strategy_infos_.size()))
    return;

  auto& info = ed_->strategy_infos_[agent_idx];
  info.agent_id = agent_idx;
  info.mode = mode;
  info.target_type = target_type;
  info.target_id = target_id;
  info.semantic_score = semantic_score;
  info.path_length = path.empty() ? -1.0 : Astar2D::pathLength(path);
  info.target_pos = target_pos;
}

double ExplorationManager::computePathCost(const Vector2d& pos1, const Vector2d& pos2)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(pos1, pos2, 0.25, 0.002) == Astar2D::REACH_END)
    return Astar2D::pathLength(path_finder_->getPath());
  return 10000.0;
}

void ExplorationManager::computeATSPCostMatrix(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& mat)
{
  int dimen = frontiers.size() + 1;
  mat.resize(dimen, dimen);

  // Agent to frontiers
  for (int i = 1; i < dimen; i++) {
    mat(0, i) = computePathCost(cur_pos, frontiers[i - 1]);
    mat(i, 0) = 0;
  }

  // Costs between frontiers
  for (int i = 1; i < dimen; ++i) {
    for (int j = i + 1; j < dimen; ++j) {
      double cost = computePathCost(frontiers[i - 1], frontiers[j - 1]);
      mat(i, j) = cost;
      mat(j, i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 100000.0;
  }
}

void ExplorationManager::computeATSPTour(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices)
{
  indices.clear();
  if (frontiers.empty()) {
    ROS_ERROR("No frontier to compute tsp!");
    return;
  }
  else if (frontiers.size() == 1) {
    indices.push_back(0);
    return;
  }
  /* change ATSP to lhk3 */
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  computeATSPCostMatrix(cur_pos, frontiers, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Initialize ATSP par file
  // Create problem file
  ofstream file(ep_->tsp_dir_ + "/atsp_tour.atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * cost_mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;
  file.open(ep_->tsp_dir_ + "/atsp_tour.par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->tsp_dir_ + "/atsp_tour.atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->tsp_dir_ + "/atsp_tour.tour\n";
  file.close();

  auto par_dir = ep_->tsp_dir_ + "/atsp_tour.atsp";

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 1;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/atsp_tour.tour");
  if (!res_file.is_open()) {
    ROS_ERROR("Failed to open ATSP tour result.");
    return;
  }
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = 0;
    try {
      id = stoi(res);
    }
    catch (const std::exception&) {
      ROS_ERROR("Invalid ATSP tour entry: %s", res.c_str());
      indices.clear();
      return;
    }
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1)
      break;
    const int frontier_idx = id - 2;
    if (frontier_idx < 0 || frontier_idx >= static_cast<int>(frontiers.size())) {
      ROS_ERROR("ATSP tour index %d is outside frontier range.", frontier_idx);
      indices.clear();
      return;
    }
    indices.push_back(frontier_idx);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  // for (auto idx : indices) ROS_WARN("ATSP idx = %d", idx);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(object_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  if (kdtree.nearestKSearch(cur_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0) {
    ROS_ERROR("[Bug] No nearest object point found.");
    return Vector2d(-1000.0, -1000.0);  // Error indicator
  }

  int nearest_idx = pointIdxNKNSearch[0];
  auto nearest_point = object_cloud->points[nearest_idx];
  return Vector2d(nearest_point.x, nearest_point.y);
}

bool ExplorationManager::trySearchObjectPathWithDistance(const Vector2d& start2d,
    const Vector2d& object_pose, double distance, double max_search_time,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    const std::string& debug_msg)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, distance, max_search_time) ==
      Astar2D::REACH_END) {
    std::vector<Eigen::Vector2d> path = path_finder_->getPath();
    Vector2d tmp_pos(-1000.0, -1000.0);

    // Find valid position along the path (from end to start)
    for (int i = path.size() - 1; i >= 0; i--) {
      if (sdf_map_->getOccupancy(path[i]) != SDFMap2D::OCCUPIED &&
          sdf_map_->getOccupancy(path[i]) != SDFMap2D::UNKNOWN &&
          sdf_map_->getInflateOccupancy(path[i]) != 1) {
        tmp_pos = path[i];
        break;
      }
    }

    // Search path to the valid position
    path_finder_->reset();
    if (path_finder_->astarSearch(start2d, tmp_pos, 0.2, max_search_time) == Astar2D::REACH_END) {
      refined_path = path_finder_->getPath();
      refined_pos = tmp_pos;
      if (!debug_msg.empty()) {
        ROS_WARN("%s", debug_msg.c_str());
      }
      return true;
    }
  }
  return false;
}

bool ExplorationManager::searchObjectPath(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  constexpr double kObjectAstarMaxSearchTime = 0.4;
  const double max_search_time = kObjectAstarMaxSearchTime;  // Maximum planning time per attempt
  Vector2d start2d = Vector2d(start(0), start(1));

  // Find nearest accessible point in object cloud
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error indicator from findNearestObjectPoint

  // Try different safety distances in order of preference
  const std::vector<double> distances = { 0.5, 0.70, 0.85 };
  const std::vector<std::string> debug_messages = { "I'm going to the object! dist = 0.5m!",
    "I'm going to the object! dist = 0.70m!", "I'm going to the object! dist = 0.85m!" };

  // Attempt path planning with each safety distance
  for (size_t i = 0; i < distances.size(); ++i) {
    if (trySearchObjectPathWithDistance(start2d, object_pose, distances[i], max_search_time,
            refined_pos, refined_path, debug_messages[i])) {
      return true;
    }
  }

  ROS_ERROR("Failed to find object path.");
  return false;
}

void ExplorationManager::getSortedSemanticFrontiers(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<SemanticFrontier>& sem_frontiers)
{
  // Filter and sort frontiers based on semantic values and reachability
  sem_frontiers.clear();

  for (auto& frontier : frontiers) {
    SemanticFrontier sem_frontier;
    sem_frontier.position = frontier;

    // Compute semantic value from local neighborhood
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 grid neighborhood
    double value = sdf_map_->value_map_->getValue(idx);

    // Find maximum semantic value in neighborhood (ignoring occupied cells)
    for (auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(nbr) == 1 ||
          sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
        continue;
      value = std::max(value, sdf_map_->value_map_->getValue(nbr));
    }
    sem_frontier.semantic_value = value;

    // Validate reachability and compute path cost
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (!searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path)) {
      // Assign high cost penalty for unreachable frontiers
      sem_frontier.path_length = 1000000;
      sem_frontier.path.clear();
    }
    else {
      sem_frontier.path_length = Astar2D::pathLength(tmp_path);
      sem_frontier.path = tmp_path;
    }

    // Only include frontiers with valid paths
    if (!sem_frontier.path.empty())
      sem_frontiers.push_back(sem_frontier);
  }

  // Sort by semantic value (desc) then by path length (asc)
  std::sort(sem_frontiers.begin(), sem_frontiers.end());
}

void ExplorationManager::calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers,
    double& std_dev, double& max_to_mean, double& mean, bool if_print)
{
  // Handle empty frontier list
  if (sem_frontiers.empty()) {
    std::cout << "No semantic frontiers available." << std::endl;
    max_to_mean = 1.0;  // Neutral ratio
    std_dev = 0.0;      // No variation
    return;
  }

  // Compute mean and maximum semantic values
  double sum = 0.0;
  double max_value = 0.0;
  for (const auto& frontier : sem_frontiers) {
    sum += frontier.semantic_value;
    max_value = max(max_value, frontier.semantic_value);
  }
  mean = sum / sem_frontiers.size();

  // Compute standard deviation
  double variance_sum = 0.0;
  for (const auto& frontier : sem_frontiers)
    variance_sum += (frontier.semantic_value - mean) * (frontier.semantic_value - mean);

  max_to_mean = max_value / mean;
  std_dev = std::sqrt(variance_sum / sem_frontiers.size());

  // Print summary statistics
  std::cout << "Mean Value: " << std::fixed << std::setprecision(3) << mean;
  std::cout << " , Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev;
  std::cout << " , Max-to-Mean: " << std::fixed << std::setprecision(3) << max_to_mean << std::endl;

  // Print detailed frontier values if requested
  if (if_print) {
    for (const auto& sem_frontier : sem_frontiers)
      std::cout << "Value: " << std::fixed << std::setprecision(3) << sem_frontier.semantic_value
                << std::endl;
  }
}

bool ExplorationManager::planTrajectory(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl)
{
  if (!gcopter_ || !kinoastar_) {
    ROS_WARN_THROTTLE(1.0, "[ExplorationManager] GCopter or KinoAstar not initialized for "
                           "real-world mode");
    return false;
  }

  Eigen::VectorXd goal_state, current_state;
  Vector3d control = ctrl;
  goal_state = end;
  current_state = start;

  // Kinodynamic A* search
  kinoastar_->reset();
  kinoastar_->search(goal_state, current_state, control);
  kinoastar_->getKinoNode();

  if (kinoastar_->has_path_) {
    kinoastar_->kinoastarFlatPathPub(kinoastar_->flat_trajs_);
    gcopter_->minco_plan();
    std::vector<Trajectory<7, 3>> final_trajes = gcopter_->final_trajes;
    gcopter_->mincoPathPub(gcopter_->final_trajes, gcopter_->final_singuls);
    return true;
  }

  return false;
}

}  // namespace apexnav_planner
