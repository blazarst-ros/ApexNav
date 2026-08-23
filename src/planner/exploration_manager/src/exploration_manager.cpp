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
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <cstdio>
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

  nh.param("multi_agent/voronoi/enabled", voronoi_enabled_, true);
  nh.param("multi_agent/voronoi/soft_fallback", voronoi_soft_fallback_, true);
  nh.param("multi_agent/voronoi/debug", voronoi_debug_, false);
  nh.param("multi_agent/voronoi/semantic_weight", voronoi_config_.semantic_weight, 1.0);
  nh.param("multi_agent/voronoi/distance_scale", voronoi_config_.distance_scale_m, 10.0);
  nh.param("multi_agent/voronoi/distance_cap", voronoi_config_.distance_cap_m, 20.0);
  nh.param("multi_agent/voronoi/max_bias", voronoi_config_.max_bias_m, 3.0);
  nh.param("multi_agent/voronoi/owner_hysteresis", voronoi_config_.owner_hysteresis_m, 0.3);
  nh.param("multi_agent/voronoi/balance_gain", voronoi_config_.balance_gain_m, 1.0);
  nh.param("multi_agent/voronoi/max_balance_iterations",
      voronoi_config_.max_balance_iterations, 5);
  nh.param("multi_agent/voronoi/movement_trigger", voronoi_movement_trigger_, 1.0);
  nh.param("multi_agent/voronoi/min_repartition_period",
      voronoi_min_repartition_period_, 1.0);
  nh.param("multi_agent/voronoi/visualization_resolution",
      voronoi_visualization_resolution_, 0.25);
  nh.param("multi_agent/voronoi/frontier_region_radius",
      voronoi_frontier_region_radius_, 3.0);
  voronoi_allocator_.reset(new DynamicVoronoiAllocator(voronoi_config_));
  voronoi_region_pub_ =
      nh.advertise<sensor_msgs::PointCloud2>("/multi_agent/voronoi_regions", 1, true);

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
  if (voronoi_allocator_)
    voronoi_allocator_->reset();
  voronoi_grid_ = VoronoiGrid();
  voronoi_result_ = VoronoiResult();
  voronoi_frontier_owner_by_address_.clear();
  last_voronoi_agent_positions_.clear();
  last_voronoi_agent_active_.clear();
  last_voronoi_frontier_signature_ = 0;
  last_voronoi_update_ = ros::Time();
}

size_t ExplorationManager::frontierSignature() const
{
  size_t signature = ed_->frontier_averages_.size() * 1315423911u +
                     ed_->dormant_frontier_averages_.size();
  auto mix = [&](const Vector2d& frontier) {
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    const size_t value = static_cast<size_t>(sdf_map_->toAddress(idx));
    signature ^= value + 0x9e3779b9 + (signature << 6) + (signature >> 2);
  };
  for (const auto& frontier : ed_->frontier_averages_)
    mix(frontier);
  for (const auto& frontier : ed_->dormant_frontier_averages_)
    mix(frontier);
  return signature;
}

VoronoiGrid ExplorationManager::buildVoronoiGrid(
    const vector<Vector2d>& active_frontiers) const
{
  VoronoiGrid grid;
  Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  grid.resolution = sdf_map_->getResolution();
  const int map_width = std::max(1, static_cast<int>(std::ceil(size.x() / grid.resolution)));
  const int map_height = std::max(1, static_cast<int>(std::ceil(size.y() / grid.resolution)));
  const bool use_frontier_region = !active_frontiers.empty();
  const int region_radius_cells = std::max(
      0, static_cast<int>(std::ceil(voronoi_frontier_region_radius_ / grid.resolution)));
  int min_x = use_frontier_region ? map_width : 0;
  int min_y = use_frontier_region ? map_height : 0;
  int max_x = use_frontier_region ? -1 : map_width - 1;
  int max_y = use_frontier_region ? -1 : map_height - 1;
  if (use_frontier_region) {
    for (const auto& frontier : active_frontiers) {
      Vector2i frontier_index;
      sdf_map_->posToIndex(frontier, frontier_index);
      min_x = std::min(min_x, std::max(0, frontier_index.x() - region_radius_cells));
      min_y = std::min(min_y, std::max(0, frontier_index.y() - region_radius_cells));
      max_x = std::max(max_x, std::min(map_width - 1, frontier_index.x() + region_radius_cells));
      max_y = std::max(max_y, std::min(map_height - 1, frontier_index.y() + region_radius_cells));
    }
  }
  if (max_x < min_x || max_y < min_y) {
    grid.width = 1;
    grid.height = 1;
    grid.traversable.assign(1, 0);
    return grid;
  }
  grid.offset_x = min_x;
  grid.offset_y = min_y;
  grid.width = max_x - min_x + 1;
  grid.height = max_y - min_y + 1;
  grid.traversable.assign(grid.width * grid.height, 0);
  for (int x = 0; x < grid.width; ++x) {
    for (int y = 0; y < grid.height; ++y) {
      const Vector2i idx(x + grid.offset_x, y + grid.offset_y);
      if (sdf_map_->getOccupancy(idx) != SDFMap2D::FREE ||
          sdf_map_->getInflateOccupancy(idx) == 1)
        continue;
      if (!use_frontier_region) {
        grid.traversable[x * grid.height + y] = 1;
        continue;
      }
      Vector2d position;
      sdf_map_->indexToPos(idx, position);
      for (const auto& frontier : active_frontiers) {
        if ((position - frontier).squaredNorm() <=
            voronoi_frontier_region_radius_ * voronoi_frontier_region_radius_) {
          grid.traversable[x * grid.height + y] = 1;
          break;
        }
      }
    }
  }
  return grid;
}

bool ExplorationManager::projectFrontierCluster(const vector<Vector2d>& cluster,
    const Vector2d& average, Vector2i& projected_index) const
{
  double best_squared_distance = std::numeric_limits<double>::infinity();
  static const int offsets[8][2] = {
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  for (const auto& cell : cluster) {
    Vector2i frontier_index;
    sdf_map_->posToIndex(cell, frontier_index);
    for (const auto& offset : offsets) {
      const Vector2i candidate = frontier_index + Vector2i(offset[0], offset[1]);
      if (!sdf_map_->isInMap(candidate) ||
          sdf_map_->getOccupancy(candidate) != SDFMap2D::FREE ||
          sdf_map_->getInflateOccupancy(candidate) == 1)
        continue;
      Vector2d position;
      sdf_map_->indexToPos(candidate, position);
      const double squared_distance = (position - average).squaredNorm();
      if (squared_distance < best_squared_distance) {
        best_squared_distance = squared_distance;
        projected_index = candidate;
      }
    }
  }
  return std::isfinite(best_squared_distance);
}

void ExplorationManager::publishVoronoiRegions(
    const VoronoiGrid& grid, const VoronoiResult& result)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  const int stride = std::max(
      1, static_cast<int>(std::ceil(voronoi_visualization_resolution_ / grid.resolution)));
  cloud.points.reserve(result.owner_by_cell.size() / (stride * stride) + 1);
  for (int x = 0; x < grid.width; x += stride) {
    for (int y = 0; y < grid.height; y += stride) {
      const int adr = x * grid.height + y;
      if (adr >= static_cast<int>(result.owner_by_cell.size()) || result.owner_by_cell[adr] < 0)
        continue;
      Vector2d position;
      sdf_map_->indexToPos(
          Vector2i(x + grid.offset_x, y + grid.offset_y), position);
      pcl::PointXYZRGB point;
      point.x = position.x();
      point.y = position.y();
      point.z = 0.05;
      const VoronoiDisplayColor color = voronoiDisplayColor(result.owner_by_cell[adr]);
      point.r = color.r;
      point.g = color.g;
      point.b = color.b;
      cloud.points.push_back(point);
    }
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  sensor_msgs::PointCloud2 message;
  pcl::toROSMsg(cloud, message);
  message.header.frame_id = "world";
  message.header.stamp = ros::Time::now();
  voronoi_region_pub_.publish(message);
}

void ExplorationManager::updateVoronoiAllocation(const vector<Vector2d>& agent_positions,
    const vector<bool>& agent_active, bool frontier_changed)
{
  if (!voronoi_enabled_ || !voronoi_allocator_ || agent_positions.size() != agent_active.size())
    return;

  const size_t signature = frontierSignature();
  const bool active_changed = agent_active != last_voronoi_agent_active_;
  bool moved = agent_positions.size() != last_voronoi_agent_positions_.size();
  if (!moved) {
    for (int i = 0; i < static_cast<int>(agent_positions.size()); ++i) {
      if (agent_active[i] &&
          (agent_positions[i] - last_voronoi_agent_positions_[i]).norm() >=
              voronoi_movement_trigger_) {
        moved = true;
        break;
      }
    }
  }
  const bool no_previous = voronoi_result_.owner_by_cell.empty();
  const bool frontier_set_changed = frontier_changed || signature != last_voronoi_frontier_signature_;
  if (!no_previous && !active_changed && !moved && !frontier_set_changed)
    return;

  std::string update_reason;
  const auto append_reason = [&](const std::string& reason) {
    if (!update_reason.empty())
      update_reason += "+";
    update_reason += reason;
  };
  if (no_previous)
    append_reason("initial");
  if (active_changed)
    append_reason("activity_changed");
  if (moved)
    append_reason("agent_moved");
  if (frontier_set_changed)
    append_reason("frontier_changed");

  const ros::Time now = ros::Time::now();
  if (!no_previous && !active_changed && !last_voronoi_update_.isZero() &&
      (now - last_voronoi_update_).toSec() < voronoi_min_repartition_period_)
    return;

  const ros::WallTime begin = ros::WallTime::now();
  VoronoiGrid grid = buildVoronoiGrid(ed_->frontier_averages_);
  vector<VoronoiAgent> agents;
  for (int i = 0; i < static_cast<int>(agent_positions.size()); ++i) {
    Vector2i idx;
    sdf_map_->posToIndex(agent_positions[i], idx);
    agents.push_back(
        {i, idx.x() - grid.offset_x, idx.y() - grid.offset_y, agent_active[i]});
  }

  vector<VoronoiFrontier> frontiers;
  vector<int> frontier_addresses;
  int frontier_id = 0;
  auto append_frontiers =
      [&](const vector<Vector2d>& positions, const vector<vector<Vector2d>>& clusters,
          bool active) {
    const int count = static_cast<int>(std::min(positions.size(), clusters.size()));
    for (int i = 0; i < count; ++i) {
      Vector2i projected_index;
      if (!projectFrontierCluster(clusters[i], positions[i], projected_index))
        continue;
      Vector2i original_index;
      sdf_map_->posToIndex(positions[i], original_index);
      frontiers.push_back({frontier_id++, projected_index.x() - grid.offset_x,
          projected_index.y() - grid.offset_y, getFrontierSemanticValue(positions[i]), active});
      frontier_addresses.push_back(sdf_map_->toAddress(original_index));
    }
  };
  append_frontiers(ed_->frontier_averages_, ed_->frontiers_, true);
  append_frontiers(ed_->dormant_frontier_averages_, ed_->dormant_frontiers_, false);

  VoronoiResult result = voronoi_allocator_->allocate(grid, agents, frontiers);
  voronoi_frontier_owner_by_address_.clear();
  for (int i = 0; i < static_cast<int>(frontiers.size()); ++i) {
    voronoi_frontier_owner_by_address_[frontier_addresses[i]] = result.frontier_owner[i];
  }
  voronoi_grid_ = std::move(grid);
  voronoi_result_ = std::move(result);
  last_voronoi_agent_positions_ = agent_positions;
  last_voronoi_agent_active_ = agent_active;
  last_voronoi_frontier_signature_ = signature;
  last_voronoi_update_ = now;
  publishVoronoiRegions(voronoi_grid_, voronoi_result_);

  vector<int> cell_counts(agent_positions.size(), 0);
  vector<int> frontier_counts(agent_positions.size(), 0);
  for (const int owner : voronoi_result_.owner_by_cell)
    if (owner >= 0 && owner < static_cast<int>(cell_counts.size()))
      ++cell_counts[owner];
  for (const int owner : voronoi_result_.frontier_owner)
    if (owner >= 0 && owner < static_cast<int>(frontier_counts.size()))
      ++frontier_counts[owner];
  const double elapsed_ms = (ros::WallTime::now() - begin).toSec() * 1000.0;
  ROS_INFO("[DynamicVoronoi] elapsed_ms=%.2f cells=[%d,%d] frontiers=[%d,%d] "
           "load=[%.2f,%.2f] bias=[%.2f,%.2f] reassigned_active_frontiers=%d",
      elapsed_ms, cell_counts.size() > 0 ? cell_counts[0] : 0,
      cell_counts.size() > 1 ? cell_counts[1] : 0,
      frontier_counts.size() > 0 ? frontier_counts[0] : 0,
      frontier_counts.size() > 1 ? frontier_counts[1] : 0,
      voronoi_result_.loads.size() > 0 ? voronoi_result_.loads[0] : 0.0,
      voronoi_result_.loads.size() > 1 ? voronoi_result_.loads[1] : 0.0,
      voronoi_result_.biases.size() > 0 ? voronoi_result_.biases[0] : 0.0,
      voronoi_result_.biases.size() > 1 ? voronoi_result_.biases[1] : 0.0,
      voronoi_result_.enforced_frontier_reassignments);

  const int traversable_cells =
      static_cast<int>(std::count(voronoi_grid_.traversable.begin(),
          voronoi_grid_.traversable.end(), static_cast<unsigned char>(1)));
  if (voronoi_debug_) {
    ROS_INFO("[DynamicVoronoi] reason=%s grid=%dx%d offset=(%d,%d) traversable=%d "
             "active_frontiers=%zu dormant_frontiers=%zu radius=%.2f",
        update_reason.c_str(), voronoi_grid_.width, voronoi_grid_.height,
        voronoi_grid_.offset_x, voronoi_grid_.offset_y, traversable_cells,
        ed_->frontier_averages_.size(), ed_->dormant_frontier_averages_.size(),
        voronoi_frontier_region_radius_);
    for (int i = 0; i < static_cast<int>(agents.size()); ++i) {
      const int seed_address = i < static_cast<int>(voronoi_result_.seed_addresses.size())
                                   ? voronoi_result_.seed_addresses[i]
                                   : -1;
      const int effective_x = seed_address < 0 ? -1 : seed_address / voronoi_grid_.height;
      const int effective_y = seed_address < 0 ? -1 : seed_address % voronoi_grid_.height;
      const bool seed_projected = i < static_cast<int>(voronoi_result_.seed_projected.size()) &&
                                  voronoi_result_.seed_projected[i];
      ROS_INFO("[DynamicVoronoi] agent=%d active=%d pos=(%.2f,%.2f) raw_cell=(%d,%d) "
               "effective_cell=(%d,%d) projected=%d cells=%d frontiers=%d load=%.2f bias=%.2f",
          agents[i].id, agents[i].active, agent_positions[i].x(), agent_positions[i].y(),
          agents[i].x + voronoi_grid_.offset_x, agents[i].y + voronoi_grid_.offset_y,
          effective_x < 0 ? -1 : effective_x + voronoi_grid_.offset_x,
          effective_y < 0 ? -1 : effective_y + voronoi_grid_.offset_y, seed_projected,
          i < static_cast<int>(cell_counts.size()) ? cell_counts[i] : 0,
          i < static_cast<int>(frontier_counts.size()) ? frontier_counts[i] : 0,
          i < static_cast<int>(voronoi_result_.loads.size()) ? voronoi_result_.loads[i] : 0.0,
          i < static_cast<int>(voronoi_result_.biases.size()) ? voronoi_result_.biases[i] : 0.0);
    }
  }
  if (traversable_cells == 0)
    ROS_WARN("[DynamicVoronoi] active frontier region contains no traversable cells");
  int active_agent_count = 0;
  for (const auto& agent : agents)
    if (agent.active)
      ++active_agent_count;
  if (active_agent_count >= 2) {
    for (int i = 0; i < static_cast<int>(agents.size()); ++i) {
      if (!agents[i].active)
        continue;
      if (i >= static_cast<int>(voronoi_result_.seed_addresses.size()) ||
          voronoi_result_.seed_addresses[i] < 0)
        ROS_WARN("[DynamicVoronoi] agent %d has no usable seed", agents[i].id);
      else if (cell_counts[i] == 0)
        ROS_WARN("[DynamicVoronoi] agent %d owns no active-region cells", agents[i].id);
      else if (frontier_counts[i] == 0)
        ROS_WARN("[DynamicVoronoi] agent %d owns no projected frontiers", agents[i].id);
    }
  }
}

int ExplorationManager::getVoronoiOwner(const Vector2d& position) const
{
  if (!voronoi_enabled_ || voronoi_result_.owner_by_cell.empty())
    return -1;
  Vector2i idx;
  sdf_map_->posToIndex(position, idx);
  if (sdf_map_->isInMap(idx)) {
    const auto frontier_owner =
        voronoi_frontier_owner_by_address_.find(sdf_map_->toAddress(idx));
    if (frontier_owner != voronoi_frontier_owner_by_address_.end())
      return frontier_owner->second;
  }
  const int local_x = idx.x() - voronoi_grid_.offset_x;
  const int local_y = idx.y() - voronoi_grid_.offset_y;
  if (local_x < 0 || local_x >= voronoi_grid_.width || local_y < 0 ||
      local_y >= voronoi_grid_.height)
    return -1;
  return voronoi_result_.owner_by_cell[local_x * voronoi_grid_.height + local_y];
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
  const bool use_voronoi = voronoi_enabled_ && !voronoi_result_.owner_by_cell.empty();
  if (use_voronoi) {
    chooseVoronoiFrontierPolicy(pos2d, ed_->frontier_averages_, false,
        next_best_pos, next_best_path, agent_idx);
  }
  else {
    chooseExplorationPolicy(
        pos2d, ed_->frontier_averages_, next_best_pos, next_best_path, agent_idx);
  }

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
    if (use_voronoi) {
      // Exhaust this agent's own active and dormant region before borrowing another region.
      chooseVoronoiFrontierPolicy(pos2d, ed_->dormant_frontier_averages_, false,
          next_best_pos, next_best_path, agent_idx);
      if (next_best_path.empty() && voronoi_soft_fallback_) {
        chooseVoronoiFrontierPolicy(pos2d, ed_->frontier_averages_, true,
            next_best_pos, next_best_path, agent_idx);
        if (next_best_path.empty()) {
          chooseVoronoiFrontierPolicy(pos2d, ed_->dormant_frontier_averages_, true,
              next_best_pos, next_best_path, agent_idx);
        }
        if (!next_best_path.empty()) {
          ROS_WARN("[DynamicVoronoi] Agent %d temporarily leased a frontier outside its region",
              agent_idx);
        }
      }
    }
    else {
      // Try dormant frontiers as last resort.
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path, agent_idx);
    }

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

void ExplorationManager::chooseVoronoiFrontierPolicy(Vector2d cur_pos,
    const vector<Vector2d>& frontiers, bool fallback_region, Vector2d& next_best_pos,
    vector<Vector2d>& next_best_path, int agent_idx)
{
  vector<int> voronoi_owners;
  vector<int> claimed_by;
  voronoi_owners.reserve(frontiers.size());
  claimed_by.reserve(frontiers.size());
  for (const auto& frontier : frontiers) {
    voronoi_owners.push_back(getVoronoiOwner(frontier));
    int claim_owner = -1;
    for (int candidate_agent = 0; candidate_agent < NUM_AGENTS; ++candidate_agent) {
      if (frontier_map2d_->isFrontierClaimedByPosition(frontier, candidate_agent)) {
        claim_owner = candidate_agent;
        break;
      }
    }
    claimed_by.push_back(claim_owner);
  }

  const FrontierPartition partition =
      partitionFrontierCandidates(voronoi_owners, claimed_by, agent_idx);
  const vector<int>& selected_indices =
      fallback_region ? partition.fallback_indices : partition.owned_indices;
  vector<Vector2d> selected_frontiers;
  selected_frontiers.reserve(selected_indices.size());
  for (const int index : selected_indices)
    selected_frontiers.push_back(frontiers[index]);
  runVoronoiFrontierPolicy(
      cur_pos, selected_frontiers, next_best_pos, next_best_path, agent_idx);
}

void ExplorationManager::runVoronoiFrontierPolicy(Vector2d cur_pos,
    const vector<Vector2d>& frontiers, Vector2d& next_best_pos,
    vector<Vector2d>& next_best_path, int agent_idx)
{
  next_best_path.clear();
  if (frontiers.empty())
    return;
  if (frontiers.size() == 1) {
    if (searchFrontierPath(cur_pos, frontiers.front(), next_best_pos, next_best_path)) {
      setStrategyInfo(agent_idx, "VORONOI_FRONTIER", "FRONTIER",
          findFrontierIdByPosition(next_best_pos), getFrontierSemanticValue(next_best_pos),
          next_best_path, next_best_pos);
    }
    return;
  }
  findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path, agent_idx);
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
  computeATSPTour(cur_pos, filter_frontiers, indices, agent_idx);
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
  if (next_best_path.empty() && !filter_frontiers.empty()) {
    ROS_WARN("Agent %d: ATSP unavailable; keeping Voronoi ownership and using closest reachable frontier",
        agent_idx);
    findClosestFrontierPolicy(
        cur_pos, filter_frontiers, next_best_pos, next_best_path, agent_idx);
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

void ExplorationManager::computeATSPTour(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<int>& indices, int agent_idx)
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
  const string file_stem = agentAtspStem(agent_idx);
  const string problem_file = ep_->tsp_dir_ + "/" + file_stem + ".atsp";
  const string parameter_file = ep_->tsp_dir_ + "/" + file_stem + ".par";
  const string tour_file = ep_->tsp_dir_ + "/" + file_stem + ".tour";
  ofstream file(problem_file);
  if (!file.is_open()) {
    ROS_ERROR("Failed to create ATSP problem file: %s", problem_file.c_str());
    return;
  }
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
  file.open(parameter_file);
  if (!file.is_open()) {
    ROS_ERROR("Failed to create ATSP parameter file: %s", parameter_file.c_str());
    return;
  }
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + problem_file + "\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + tour_file + "\n";
  file.close();

  // Never accept a stale tour when the solver fails to produce a fresh result.
  std::remove(tour_file.c_str());

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = atspProblemCode(agent_idx);
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  ifstream res_file(tour_file);
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
