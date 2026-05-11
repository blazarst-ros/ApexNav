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

void ExplorationManager::buildRoutingTasks(vector<RoutingTask>& tasks)
{
  tasks.clear();

  vector<VerificationCandidate> verification_candidates;
  sdf_map_->object_map2d_->getVerificationCandidates(verification_candidates);
  for (const auto& candidate : verification_candidates) {
    if (!candidate.object_cloud || candidate.object_cloud->points.empty())
      continue;
    RoutingTask task;
    task.type = MTSP_TASK_VERIFY_OBJECT;
    task.priority_bonus = 600.0;
    task.object_id = candidate.object_id;
    task.position = candidate.position;
    task.target_mu_v = candidate.target_mu_v;
    task.target_sigma_v = candidate.target_sigma_v;
    task.verification_margin = candidate.margin;
    task.source_agent_id = candidate.source_agent_id;
    task.object_cloud = candidate.object_cloud;
    tasks.push_back(task);
  }

  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> strict_object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(strict_object_clouds);
  for (const auto& cloud : strict_object_clouds) {
    if (!cloud || cloud->points.empty())
      continue;
    RoutingTask task;
    task.type = MTSP_TASK_STRICT_OBJECT;
    task.priority_bonus = 1000.0;
    task.object_cloud = cloud;
    task.position.setZero();
    for (const auto& p : cloud->points) {
      task.position(0) += p.x;
      task.position(1) += p.y;
    }
    task.position /= double(cloud->points.size());
    tasks.push_back(task);
  }

  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> suspicious_object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(suspicious_object_clouds, false);
  for (const auto& cloud : suspicious_object_clouds) {
    if (!cloud || cloud->points.empty())
      continue;
    RoutingTask task;
    task.type = MTSP_TASK_SUSPICIOUS_OBJECT;
    task.priority_bonus = 250.0;
    task.object_cloud = cloud;
    task.position.setZero();
    for (const auto& p : cloud->points) {
      task.position(0) += p.x;
      task.position(1) += p.y;
    }
    task.position /= double(cloud->points.size());
    bool duplicate_object = false;
    for (const auto& existing : tasks) {
      if ((existing.type == MTSP_TASK_STRICT_OBJECT ||
              existing.type == MTSP_TASK_SUSPICIOUS_OBJECT) &&
          (existing.position - task.position).norm() < 0.5) {
        duplicate_object = true;
        break;
      }
    }
    if (duplicate_object)
      continue;
    tasks.push_back(task);
  }

  for (const auto& frontier : ed_->frontier_averages_) {
    RoutingTask task;
    task.type = MTSP_TASK_FRONTIER;
    task.position = frontier;

    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);
    double semantic_value = sdf_map_->value_map_->getValue(idx);
    for (const auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(nbr) == 1 ||
          sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
        continue;
      semantic_value = std::max(semantic_value, sdf_map_->value_map_->getValue(nbr));
    }
    task.priority_bonus = std::min(100.0, semantic_value * 100.0);
    tasks.push_back(task);
  }
}

bool ExplorationManager::refineTaskPath(const Vector3d& start, const RoutingTask& task,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  if ((task.type == MTSP_TASK_STRICT_OBJECT || task.type == MTSP_TASK_VERIFY_OBJECT ||
          task.type == MTSP_TASK_SUSPICIOUS_OBJECT) &&
      task.object_cloud && !task.object_cloud->points.empty()) {
    return searchObjectPath(start, task.object_cloud, refined_pos, refined_path);
  }

  return searchFrontierPath(Vector2d(start(0), start(1)), task.position, refined_pos, refined_path);
}

void ExplorationManager::planMultiAgentAssignments(const vector<Vector2d>& agent_positions,
    const vector<double>& agent_heights, const vector<bool>& active_agents)
{
  constexpr int TOP_K_REFINE = 5;
  constexpr double LOCAL_FRONTIER_RADIUS = 4.0;
  constexpr double HYSTERESIS_KEEP_RATIO = 0.80;
  constexpr double SAME_TASK_DISTANCE = 0.75;
  constexpr double UNREACHABLE_COST = 9999.0;

  if ((int)ed_->mtsp_tours_.size() != NUM_AGENTS) {
    ed_->mtsp_tours_.assign(NUM_AGENTS, std::vector<Vector2d>());
    ed_->mtsp_assigned_task_pos_.assign(NUM_AGENTS, Vector2d(0, 0));
    ed_->mtsp_assigned_task_type_.assign(NUM_AGENTS, -1);
    ed_->mtsp_assignment_valid_.assign(NUM_AGENTS, false);
  }

  vector<Vector2d> prev_task_pos = ed_->mtsp_assigned_task_pos_;
  vector<int> prev_task_type = ed_->mtsp_assigned_task_type_;
  vector<bool> prev_task_valid = ed_->mtsp_assignment_valid_;

  for (int i = 0; i < NUM_AGENTS; ++i) {
    ed_->mtsp_tours_[i].clear();
    ed_->mtsp_assigned_task_pos_[i] = Vector2d(0, 0);
    ed_->mtsp_assigned_task_type_[i] = -1;
    ed_->mtsp_assignment_valid_[i] = false;
  }

  vector<RoutingTask> tasks;
  buildRoutingTasks(tasks);
  if (tasks.empty() || agent_positions.empty())
    return;

  vector<int> active_indices;
  for (int i = 0; i < NUM_AGENTS && i < (int)agent_positions.size() &&
                  i < (int)active_agents.size();
       ++i) {
    if (active_agents[i])
      active_indices.push_back(i);
  }
  if (active_indices.empty())
    return;

  auto taskTypeRank = [](int type) {
    if (type == MTSP_TASK_STRICT_OBJECT)
      return 0;
    if (type == MTSP_TASK_VERIFY_OBJECT)
      return 1;
    if (type == MTSP_TASK_SUSPICIOUS_OBJECT)
      return 2;
    return 3;
  };

  auto semanticBonus = [](const RoutingTask& task) {
    if (task.type == MTSP_TASK_FRONTIER)
      return std::min(1.5, task.priority_bonus * 0.02);
    if (task.type == MTSP_TASK_VERIFY_OBJECT)
      return 0.75;
    if (task.type == MTSP_TASK_SUSPICIOUS_OBJECT)
      return 0.5;
    return 0.0;
  };

  auto verificationFitness = [&](int agent_idx, const RoutingTask& task, double distance) {
    constexpr double HEIGHT_WEIGHT = 0.75;
    constexpr double DISTANCE_DECAY = 0.25;
    const double agent_height =
        agent_idx < (int)agent_heights.size() ? agent_heights[agent_idx] : task.target_mu_v;
    const double sigma = std::max(1e-3, task.target_sigma_v);
    const double height_diff = agent_height - task.target_mu_v;
    const double height_fitness = std::exp(-(height_diff * height_diff) / (2.0 * sigma * sigma));
    const double distance_fitness = std::exp(-DISTANCE_DECAY * std::max(0.0, distance));
    return HEIGHT_WEIGHT * height_fitness + (1.0 - HEIGHT_WEIGHT) * distance_fitness;
  };
  auto verificationBonus = [&](int agent_idx, const RoutingTask& task, double distance) {
    constexpr double VERIFY_FITNESS_BONUS = 5.0;
    if (task.type != MTSP_TASK_VERIFY_OBJECT)
      return 0.0;
    return VERIFY_FITNESS_BONUS * verificationFitness(agent_idx, task, distance);
  };

  vector<vector<int>> regions(NUM_AGENTS);
  for (int task_idx = 0; task_idx < (int)tasks.size(); ++task_idx) {
    int nearest_agent = -1;
    double nearest_dist = std::numeric_limits<double>::infinity();
    double best_verify_fitness = -1.0;
    for (int agent_idx : active_indices) {
      if (tasks[task_idx].type == MTSP_TASK_VERIFY_OBJECT &&
          tasks[task_idx].source_agent_id == agent_idx)
        continue;
      double dist = (tasks[task_idx].position - agent_positions[agent_idx]).norm();
      if (tasks[task_idx].type == MTSP_TASK_VERIFY_OBJECT) {
        const double fitness = verificationFitness(agent_idx, tasks[task_idx], dist);
        if (fitness > best_verify_fitness + 1e-6 ||
            (std::fabs(fitness - best_verify_fitness) <= 1e-6 && dist < nearest_dist)) {
          best_verify_fitness = fitness;
          nearest_dist = dist;
          nearest_agent = agent_idx;
        }
        continue;
      }
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest_agent = agent_idx;
      }
    }
    if (nearest_agent >= 0)
      regions[nearest_agent].push_back(task_idx);
  }

  vector<double> selected_costs(NUM_AGENTS, UNREACHABLE_COST);
  for (int agent_idx : active_indices) {
    auto candidates = regions[agent_idx];
    if (candidates.empty()) {
      for (int task_idx = 0; task_idx < (int)tasks.size(); ++task_idx)
        candidates.push_back(task_idx);
    }

    bool has_local_frontier = false;
    for (int task_idx : candidates) {
      if (tasks[task_idx].type == MTSP_TASK_FRONTIER &&
          (tasks[task_idx].position - agent_positions[agent_idx]).norm() <=
              LOCAL_FRONTIER_RADIUS) {
        has_local_frontier = true;
        break;
      }
    }
    if (has_local_frontier) {
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                           [&](int task_idx) {
                             return tasks[task_idx].type == MTSP_TASK_FRONTIER &&
                                    (tasks[task_idx].position - agent_positions[agent_idx]).norm() >
                                        LOCAL_FRONTIER_RADIUS;
                           }),
          candidates.end());
    }

    std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
      int lhs_rank = taskTypeRank(tasks[lhs].type);
      int rhs_rank = taskTypeRank(tasks[rhs].type);
      if (lhs_rank != rhs_rank)
        return lhs_rank < rhs_rank;
      double lhs_dist = (tasks[lhs].position - agent_positions[agent_idx]).norm();
      double rhs_dist = (tasks[rhs].position - agent_positions[agent_idx]).norm();
      double lhs_score = lhs_dist - semanticBonus(tasks[lhs]);
      double rhs_score = rhs_dist - semanticBonus(tasks[rhs]);
      lhs_score -= verificationBonus(agent_idx, tasks[lhs], lhs_dist);
      rhs_score -= verificationBonus(agent_idx, tasks[rhs], rhs_dist);
      if (std::fabs(lhs_score - rhs_score) > 1e-3)
        return lhs_score < rhs_score;
      return lhs < rhs;
    });

    int best_task = -1;
    double best_score = std::numeric_limits<double>::infinity();
    double best_cost = UNREACHABLE_COST;
    const int refine_count = std::min(TOP_K_REFINE, (int)candidates.size());
    for (int i = 0; i < refine_count; ++i) {
      int task_idx = candidates[i];
      if (tasks[task_idx].type == MTSP_TASK_VERIFY_OBJECT &&
          tasks[task_idx].source_agent_id == agent_idx)
        continue;
      double path_cost = computePathCost(agent_positions[agent_idx], tasks[task_idx].position);
      if (path_cost >= UNREACHABLE_COST)
        continue;
      double score = taskTypeRank(tasks[task_idx].type) * 100.0 + path_cost -
                     semanticBonus(tasks[task_idx]);
      score -= verificationBonus(agent_idx, tasks[task_idx], path_cost);
      if (score < best_score) {
        best_score = score;
        best_task = task_idx;
        best_cost = path_cost;
      }
    }

    if ((int)prev_task_valid.size() > agent_idx &&
        (int)prev_task_type.size() > agent_idx &&
        (int)prev_task_pos.size() > agent_idx &&
        prev_task_valid[agent_idx]) {
      int prev_match = -1;
      for (int task_idx = 0; task_idx < (int)tasks.size(); ++task_idx) {
        if (tasks[task_idx].type == prev_task_type[agent_idx] &&
            (tasks[task_idx].position - prev_task_pos[agent_idx]).norm() < SAME_TASK_DISTANCE) {
          prev_match = task_idx;
          break;
        }
      }
      if (prev_match >= 0) {
        double prev_cost = computePathCost(agent_positions[agent_idx], tasks[prev_match].position);
        if (prev_cost < UNREACHABLE_COST &&
            (best_task < 0 || prev_cost <= best_cost / HYSTERESIS_KEEP_RATIO)) {
          best_task = prev_match;
          best_cost = prev_cost;
        }
      }
    }

    ed_->mtsp_tours_[agent_idx].push_back(agent_positions[agent_idx]);
    for (int task_idx : candidates)
      ed_->mtsp_tours_[agent_idx].push_back(tasks[task_idx].position);

    if (best_task >= 0) {
      ed_->mtsp_assigned_task_pos_[agent_idx] = tasks[best_task].position;
      ed_->mtsp_assigned_task_type_[agent_idx] = tasks[best_task].type;
      ed_->mtsp_assignment_valid_[agent_idx] = true;
      selected_costs[agent_idx] = best_cost;
    }
  }

  ROS_WARN("[Voronoi] Assigned local tasks across %zu active agents. Costs: %.2f %.2f %.2f",
      active_indices.size(), selected_costs[0], selected_costs[1], selected_costs[2]);
}

bool ExplorationManager::consumeAssignedTask(const Vector3d& pos, int agent_idx,
    Eigen::Vector2d& out_next_pos, std::vector<Eigen::Vector2d>& out_next_best_path, int& result)
{
  if (agent_idx < 0 || agent_idx >= NUM_AGENTS ||
      agent_idx >= (int)ed_->mtsp_assignment_valid_.size() ||
      !ed_->mtsp_assignment_valid_[agent_idx])
    return false;

  vector<RoutingTask> tasks;
  buildRoutingTasks(tasks);
  if (tasks.empty())
    return false;

  const Vector2d assigned_pos = ed_->mtsp_assigned_task_pos_[agent_idx];
  const int assigned_type = ed_->mtsp_assigned_task_type_[agent_idx];
  int best_task = -1;
  double best_dist = std::numeric_limits<double>::infinity();
  for (int i = 0; i < (int)tasks.size(); ++i) {
    if (tasks[i].type != assigned_type)
      continue;
    double dist = (tasks[i].position - assigned_pos).norm();
    if (dist < best_dist) {
      best_dist = dist;
      best_task = i;
    }
  }
  if (best_task < 0 && assigned_type == MTSP_TASK_VERIFY_OBJECT) {
    if (!searchFrontierPath(Vector2d(pos(0), pos(1)), assigned_pos, out_next_pos,
            out_next_best_path))
      return false;
    result = SEARCH_VERIFY_OBJECT;
    return true;
  }
  if (best_task < 0 || best_dist > 0.75)
    return false;

  if (!refineTaskPath(pos, tasks[best_task], out_next_pos, out_next_best_path))
    return false;

  if (assigned_type == MTSP_TASK_STRICT_OBJECT)
    result = SEARCH_BEST_OBJECT;
  else if (assigned_type == MTSP_TASK_VERIFY_OBJECT)
    result = SEARCH_VERIFY_OBJECT;
  else if (assigned_type == MTSP_TASK_SUSPICIOUS_OBJECT)
    result = SEARCH_SUSPICIOUS_OBJECT;
  else
    result = EXPLORATION;
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

  int assigned_result = EXPLORATION;
  if (consumeAssignedTask(pos, agent_idx, out_next_pos, out_next_best_path, assigned_result)) {
    if (assigned_result == EXPLORATION)
      frontier_map2d_->claimFrontierByPosition(out_next_pos, agent_idx);
    return assigned_result;
  }

  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // ==================== Navigation Mode: High-Confidence Objects ====================

  if (!object_clouds.empty()) {  // 存在高置信度目标物体点云
    ROS_WARN("[Agent %d Navigation Mode] Get object_cloud num = %ld", agent_idx, object_clouds.size());

    // Try to find path to each detected object in order of confidence
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, out_next_pos, out_next_best_path))
        return SEARCH_BEST_OBJECT;
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    ROS_WARN("[Agent %d Navigation Mode (Over Depth)] Get over depth object cloud", agent_idx);
    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, out_next_pos, out_next_best_path))
      return SEARCH_OVER_DEPTH_OBJECT;
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
        searchObjectPath(pos, top_object_cloud, out_next_pos, out_next_best_path))
      return SEARCH_SUSPICIOUS_OBJECT;
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path, agent_idx);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      ROS_ERROR("Agent %d: search exterme case!!!", agent_idx);

      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, out_next_pos, out_next_best_path))
          return SEARCH_EXTREME;
      }

      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, out_next_pos, out_next_best_path))
          return SEARCH_EXTREME;
      }

      static auto last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;
      if (!object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud->points.empty() &&
          searchObjectPathExtreme(
              pos, last_over_depth_object_cloud, out_next_pos, out_next_best_path)) {
        return SEARCH_EXTREME;
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        ROS_ERROR("Agent %d: No coverable frontier!!", agent_idx);
        return NO_COVERABLE_FRONTIER;
      }
      else {
        ROS_ERROR("Agent %d: No passable frontier!!", agent_idx);
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Store successful planning results
  out_next_pos = next_best_pos;
  out_next_best_path = next_best_path;

  // Claim this frontier for the agent
  frontier_map2d_->claimFrontierByPosition(next_best_pos, agent_idx);

  // Performance monitoring
  double total_time = (ros::Time::now() - t2).toSec();
  ROS_ERROR_COND(total_time > 0.25, "[Agent %d Plan NBV] Total time %.2lf s too long!!!", agent_idx, total_time);

  return EXPLORATION;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path, int agent_idx)
{
  // Filter out frontiers claimed by any other agent.
  vector<Vector2d> original_frontiers = frontiers;  // keep for fallback
  frontiers.erase(
      std::remove_if(frontiers.begin(), frontiers.end(),
          [&](const Vector2d& f) {
              for (int other_agent = 0; other_agent < NUM_AGENTS; ++other_agent) {
                if (other_agent == agent_idx)
                  continue;
                if (frontier_map2d_->isFrontierClaimedByPosition(f, other_agent))
                  return true;
              }
              return false;
          }),
      frontiers.end());

  // Fallback: if all frontiers are claimed, use unfiltered list
  if (frontiers.empty() && !original_frontiers.empty()) {
    ROS_WARN("Agent %d: All frontiers claimed by other agents, falling back to shared selection", agent_idx);
    frontiers = original_frontiers;
  }

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
  }
  else {
    ROS_WARN("Agent %d: Explore the environment (Closest)!!", agent_idx);
    findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
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
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path))
        break;
    }
  }
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
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1)
      break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  // for (auto idx : indices) ROS_WARN("ATSP idx = %d", idx);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  if (!object_cloud || object_cloud->points.empty()) {
    ROS_ERROR("[Object Path] Empty object cloud; skip object path search.");
    return Vector2d(-1000.0, -1000.0);
  }

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
  const double max_search_time = 0.2;  // Maximum planning time per attempt
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
      if (sdf_map_->getInflateOccupancy(idx) == 1 ||
          sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED)
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
