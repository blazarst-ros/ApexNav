/**
 * @file frontier_map2d.cpp
 * @brief Implementation of 2D frontier detection system for autonomous exploration
 *
 * This file provides the complete implementation of frontier detection algorithms
 * for autonomous robotic exploration. The system identifies boundaries between
 * known free space and unknown regions, performs intelligent clustering using
 * PCA-based subdivision, and provides visibility-based validation for efficient
 * exploration planning.代码实现了一个用于自主机器人探索的2D 前沿检测系统（FrontierMap2D），
 * 核心功能是识别地图中 “已知自由空间” 与 “未知区域” 的边界（前沿），
 * 并对前沿进行聚类、分割、状态管理，为机器人探索规划提供目标。
 *
 * @author Zager-Zhang
 */
#include <plan_env/frontier_map2d.h>
#include <unordered_map>

namespace apexnav_planner {
FrontierMap2D::FrontierMap2D(const SDFMap2D::Ptr& sdf_map, ros::NodeHandle& nh){
//功能：初始化前沿检测的核心依赖和参数（是FrontierMap2D的构造函数）

  // Initialize core mapping infrastructure
  this->sdf_map_ = sdf_map;//栅格地图的智能指针
  int voxel_num = sdf_map_->getVoxelNum();//栅格总数

  // Allocate and initialize frontier state flags for all grid cells
  frontier_flag_ = vector<char>(voxel_num, NONE);//标记每个栅格的前沿状态
  fill(frontier_flag_.begin(), frontier_flag_.end(), NONE);//赋值为NONE

  // Load exploration parameters from ROS parameter server从参数服务器读取指定名称的参数，赋值给类成员变量；如果参数不存在，就使用默认值()
  nh.param("frontier/cluster_min", cluster_min_, -1);//最小尺寸阈值
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);//最大尺寸阈值（米）(大于则会被PCA分割)
  nh.param("frontier/min_contain_unknown", min_contain_unknown_, 50);//前沿需要包含的最小未知栅格数（低于这个数的前沿无探索价值，会被标记为休眠）
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);//前沿的最小可视完成率

  // Initialize ray-casting system for visibility analysis
  raycaster_.reset(new RayCaster2D);
  resolution_ = sdf_map_->getResolution();//分辨率
  Eigen::Vector2d origin, size;//获取地图的原点坐标（origin） 和尺寸（size）,保证不超出范围
  sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  // Setup perception utilities for sensor integration（初始化感知工具（PerceptionUtils2D））
  percep_utils_.reset(new PerceptionUtils2D(nh));
}

void FrontierMap2D::searchFrontiers()
/*完成 “全流程的前沿检测与更新”：
先清理地图更新区域内失效的旧前沿，
再在扩展后的搜索区域内扫描并识别新的前沿种子，
通过区域生长（BFS）形成前沿聚类，对超大聚类进行 PCA 分割，
最后将新前沿整合到活跃列表并重新分配唯一ID，为机器人探索规划提供最新的、有效的前沿目标*/
{

  // Clear previous candidate frontiers from current search iteration

  candidate_frontiers_.clear();//临时存储 “新检测到的前沿” 的容器

  // Determine spatial bounds of recently updated map regions

  Vector2d update_min, update_max;
  sdf_map_->getLocalUpdatedBox(update_min, update_max);//获取地图更新区域的边界(更新地图)


  // (Defination)Lambda function for efficient frontier removal and flag reset
  auto resetFlag = [&](list<Frontier2D>::iterator& iter, list<Frontier2D>& frontiers) {
    Eigen::Vector2i idx;
    // Reset frontier flags for all cells in the frontier cluster
    for (auto cell : iter->cells_) {
      sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toAdr(idx)] = NONE;
    }
    // Remove frontier from container and return updated iterator
    iter = frontiers.erase(iter);
  };


  // Process active frontiers: remove those in updated regions if changed
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, frontiers_);
    else
      ++iter;
  }

  // Process dormant frontiers: remove those in updated regions if changed
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }
/*地图更新后，旧的前沿可能因为环境变化（如未知区域被探索、出现新障碍）而失效，清理这些前沿避免后续规划错误*/


  // Search for new frontiers within slightly expanded updated region
  Vector2d search_min = update_min - Vector2d(1, 1);
  Vector2d search_max = update_max + Vector2d(1, 1);
  Vector2d box_min, box_max;
  sdf_map_->getMapBoundary(box_min, box_max);

  // Constrain search region to valid map boundaries
  for (int k = 0; k < 2; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }

  // Convert spatial bounds to grid indices for efficient iteration (坐标转栅格索引)
  Eigen::Vector2i min_id, max_id;
  sdf_map_->posToIndex(search_min, min_id);
  sdf_map_->posToIndex(search_max, max_id);

  // Systematic grid scanning for frontier seed identification（双层循环遍历搜索区域所有栅格）
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y) {
      Eigen::Vector2i cur(x, y);

      // Check for unprocessed cells that satisfy frontier conditions
      if (frontier_flag_[toAdr(cur)] == NONE && isSatisfyFrontier(cur)) {//没有识别过且满足前沿条件
        // Initiate region growing from identified frontier seed
        expandFrontier(cur);
      }
    }

  // Apply PCA-based subdivision to large frontier clusters
  splitLargeFrontiers(candidate_frontiers_);

  // Integrate newly discovered frontiers into active frontier set
  for (auto& tmp_ftr : candidate_frontiers_) frontiers_.insert(frontiers_.end(), tmp_ftr);

  // Reassign unique identifiers to maintain frontier tracking consistency（PCA后重新分配前沿唯一 ID）
  int idx = 0;
  for (auto& ft : frontiers_) ft.id_ = idx++;
}

void FrontierMap2D::expandFrontier(const Eigen::Vector2i& first)
/*工具，
以一个 “前沿种子栅格” 为起点，通过广度优先搜索（BFS） 
扩展出连通的前沿聚类，
同时过滤掉尺寸过小的无效聚类，最终将有效聚类存入候选前沿列表，
为后续分割和使用做准备（基于 BFS 实现 “从种子到聚类” 的前沿区域生长，保证前沿的连通性）*/
{
  // Initialize data structures for breadth-first region growing
  queue<Eigen::Vector2i> cell_queue;
  vector<Eigen::Vector2d> expanded;
  Vector2d pos;

  // Add seed cell to expansion queue and mark as active
  sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toAdr(first)] = ACTIVE;

  // Execute breadth-first search for connected frontier region growing
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);

    // Examine all neighboring cells for potential cluster expansion
    for (auto nbr : nbrs) {
      int adr = toAdr(nbr);

      // Skip cells already processed or not satisfying frontier criteria
      if (frontier_flag_[adr] != NONE || !isSatisfyFrontier(nbr))
        continue;

      // Add qualified neighbor to expanding frontier cluster
      sdf_map_->indexToPos(nbr, pos);
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = ACTIVE;
    }
  }

  // Validate cluster size and create frontier object if meets minimum threshold
  if ((int)expanded.size() > cluster_min_) {
    Frontier2D frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);  // Calculate geometric properties and metadata
    candidate_frontiers_.push_back(frontier);
  }
  else {
    // Reset flags for clusters below minimum size threshold
    for (auto cell : expanded) {
      Vector2i cell_idx;
      sdf_map_->posToIndex(cell, cell_idx);
      frontier_flag_[toAdr(cell_idx)] = NONE;
    }
  }
}

/**
 * @brief Apply PCA-based subdivision to large frontier clusters for exploration optimization
 * @param frontiers Reference to frontier list for in-place modification
 */
void FrontierMap2D::splitLargeFrontiers(list<Frontier2D>& frontiers)//遍历所有前沿聚类，对每个聚类调用真正实现 PCA 分割的 splitHorizontally 函数
{
  list<Frontier2D> splits, tmps;
  //split临时存储单次 PCA 分割产生的子前沿，tmps存储所有处理后的前沿（拆分后的子前沿 + 无需拆分的原前沿）

  // Process each frontier for potential horizontal subdivision
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Attempt PCA-based horizontal splitting of current frontier
    if (splitHorizontally(*it, splits)) {
      // Integration subdivided fragments into temporary collection
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    }
    else {
      // Retain original frontier if subdivision not beneficial
      tmps.push_back(*it);
    }
  }

  // Replace original frontier list with processed results
  frontiers = tmps;
}

bool FrontierMap2D::splitHorizontally(const Frontier2D& frontier, list<Frontier2D>& splits)
{
  auto mean = frontier.average_;  // Spatial centroid for PCA analysis
  bool need_split = false;

  // Check if any frontier cells exceed the spatial clustering threshold（提前结束聚类）
  for (auto cell : frontier.cells_) {
    if ((cell - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }

  // Return early if frontier size is within acceptable bounds
  if (!need_split)
    return false;

  // Compute covariance matrix for Principal Component Analysis
  //(协方差矩阵描述了数据在不同维度上的 “离散程度”)
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.cells_) {
    Eigen::Vector2d diff = cell - mean;
    cov += diff * diff.transpose();  // Outer product for covariance computation
  }
  cov /= double(frontier.cells_.size());  // Normalize by sample count

  // Extract principal component eigenvector for optimal splitting direction
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
  Eigen::Vector2d first_pc = es.eigenvectors().col(1);  // Largest eigenvalue direction

  // Partition frontier cells along the primary principal component axis
  Frontier2D ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    // Project cell displacement onto primary component for binary classification
    if ((cell - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }

  // Compute geometric properties for both frontier subdivisions
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursively apply subdivision to first partition if still oversized
  list<Frontier2D> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  }
  else {
    splits.push_back(ftr1);
  }

  // Recursively apply subdivision to second partition if still oversized
  if (splitHorizontally(ftr2, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  }
  else {
    splits.push_back(ftr2);
  }

  return true;  // Successful subdivision completed
}

bool FrontierMap2D::dormantSeenFrontiers(Vector2d sensor_pos, double sensor_yaw)//前沿状态管理的核心函数，负责判断活跃前沿是否需要转为 “休眠（DORMANT）” 状态
{
  bool change_flag = false;//标记本次函数执行是否有前沿状态发生变化

  // Configure perception utilities with current sensor pose
  percep_utils_->setPose(sensor_pos, sensor_yaw);//后续判断 “前沿是否在传感器视野内” 需要基于这个姿态

  // Evaluate all active frontiers for potential dormancy(逐个活跃前沿评判)
  for (auto it = frontiers_.begin(); it != frontiers_.end();) {

    // Count connected unknown cells to validate frontier utility（首先过滤掉 “几乎没有未知区域” 的前沿）
    int cnt = countConnectUnknownGrids(it->average_);
    bool too_small = cnt < min_contain_unknown_;

    // Skip frontiers outside current sensor field of view（过滤传感器视野外的前沿）
    if (!percep_utils_->insideFOV(it->average_)) {
      ++it;
      continue;
    }

    // Perform ray-casting visibility analysis from sensor to frontier（射线检测判断前沿是否可视（无遮挡））
    raycaster_->input(it->average_, sensor_pos);
    Vector2i idx;
    raycaster_->nextId(idx);
    bool visib = true;

    // Trace ray path checking for occlusion by occupied cells
    while (raycaster_->nextId(idx)) {
      Vector2d pos;
      sdf_map_->indexToPos(idx, pos);

      // Skip immediate sensor vicinity to avoid self-occlusion
      if ((pos - sensor_pos).norm() < 0.1)
        break;

      // Check for obstacle blocking line of sight to frontier
      if (sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED) {
        visib = false;//前沿的 “使命” 是引导探索未知，可视则使命完成
        break;
      }
    }

    // Move frontier to dormant state if visible or too small for exploration（状态转换（活跃→休眠））
    if (visib || too_small) {
      dormant_frontiers_.push_back(*it);

      // Update frontier flags to dormant state for all constituent cells
      for (auto cell : it->cells_) {
        Vector2i idx;
        sdf_map_->posToIndex(cell, idx);
        frontier_flag_[toAdr(idx)] = DORMANT;
      }
      // Remove frontier from active list and update tracking state
      it = frontiers_.erase(it);
      change_flag = true;
    }
    else {
      ++it;  // Continue to next frontier if no state change required
    }
  }
  return change_flag;
}

bool FrontierMap2D::isFrontierChanged(const Frontier2D& ft)//判断前沿是否变化，避免重复增删
{
  // Check each cell in frontier cluster for continued boundary validity
  for (auto cell : ft.cells_) {
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(cell, idx);

    // Return true if any cell no longer satisfies frontier criteria
    if (!isSatisfyFrontier(idx))
      return true;
  }
  return false;  // All cells maintain frontier boundary properties
}

void FrontierMap2D::computeFrontierInfo(Frontier2D& ftr)
{
  // Initialize centroid accumulator and bounding box with first cell
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();

  // Accumulate spatial properties across all frontier cells
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;  // Sum for centroid calculation

    // Update axis-aligned bounding box extrema
    for (int i = 0; i < 2; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }

  // Compute final centroid as mean position of all cluster cells
  ftr.average_ /= double(ftr.cells_.size());
}

bool FrontierMap2D::isAnyFrontierChanged()
{
  // Get spatial bounds of recently updated map regions
  Vector2d update_min, update_max;
  sdf_map_->getLocalUpdatedBox(update_min, update_max);

  // Lambda function for frontier change evaluation with threshold-based detection
  auto checkChanges = [&](const list<Frontier2D>& frontiers) {
    for (auto ftr : frontiers) {
      // Skip frontiers outside updated region to optimize computation
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max))
        continue;

      // Calculate change threshold based on frontier size and configuration
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;

      // Count cells that no longer satisfy frontier boundary criteria
      for (auto cell : ftr.cells_) {
        Eigen::Vector2i idx;
        sdf_map_->posToIndex(cell, idx);

        // Increment counter and check threshold for early termination
        if (!isSatisfyFrontier(idx) && ++change_num >= change_thresh)
          return true;  // Significant changes detected
      }
    }
    return false;  // No significant changes found
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_))
    return true;
  return false;
}

int FrontierMap2D::countConnectUnknownGrids(const Eigen::Vector2d& pos)//Unkonwn区域连通性统计工具函数,这个结果用于判断前沿是否 “有探索价值”（未知栅格数是否达标）
{
  int unknown_threshold = min_contain_unknown_;

  // Initialize data structures for breadth-first connectivity search
  queue<Eigen::Vector2i> cell_queue;
  Vector2i idx;
  int cnt = 0;

  // Convert position to grid index and initialize search
  sdf_map_->posToIndex(pos, idx);
  cell_queue.push(idx);
  std::unordered_map<int, char> flag_visited;
  flag_visited[toAdr(idx)] = 1;
  cnt++;

  // Execute breadth-first search for connected unknown region analysis
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);

    // Examine all neighboring cells for unknown connectivity
    for (auto nbr : nbrs) {
      int adr = toAdr(nbr);

      // Skip cells not in unknown occupancy state or already visited
      if (sdf_map_->getOccupancy(nbr) != SDFMap2D::UNKNOWN || flag_visited.count(adr) ||
          sdf_map_->getInflateOccupancy(nbr))
        continue;

      // Add unvisited unknown cell to connectivity analysis
      cnt++;
      flag_visited[adr] = 1;
      Vector2d pos;
      sdf_map_->indexToPos(nbr, pos);
      cell_queue.push(nbr);
    }

    // Early termination if sufficient unknown cells identified
    if (cnt >= unknown_threshold)
      break;
  }
  return cnt;
}

void FrontierMap2D::setForceDormantFrontier(const Vector2d& frontier_center)//将前沿休眠
{
  // Initialize data structures for region growing around specified center
  queue<Eigen::Vector2i> cell_queue;
  vector<Eigen::Vector2d> expanded;
  Vector2i idx;

  // Convert center position to grid index and mark as force dormant
  sdf_map_->posToIndex(frontier_center, idx);
  expanded.push_back(frontier_center);
  cell_queue.push(idx);
  frontier_flag_[toAdr(idx)] = FORCE_DORMANT;

  // Execute breadth-first search for connected frontier region identification
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);

    // Examine neighbors for active or dormant frontier cells
    for (auto nbr : nbrs) {
      int adr = toAdr(nbr);

      // Only process cells that are currently active or dormant frontiers
      if (frontier_flag_[adr] != ACTIVE && frontier_flag_[adr] != DORMANT)
        continue;

      // Add frontier cell to force dormant region
      Vector2d pos;
      sdf_map_->indexToPos(nbr, pos);
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = FORCE_DORMANT;
    }
  }

  // Remove force dormant frontiers from active frontier list
  for (auto it = frontiers_.begin(); it != frontiers_.end();) {
    Vector2i avg_idx;
    sdf_map_->posToIndex(it->average_, avg_idx);
    if (frontier_flag_[toAdr(avg_idx)] == FORCE_DORMANT) {
      it = frontiers_.erase(it);
    }
    else {
      ++it;
    }
  }

  // Remove force dormant frontiers from dormant frontier list
  for (auto it = dormant_frontiers_.begin(); it != dormant_frontiers_.end();) {
    Vector2i avg_idx;
    sdf_map_->posToIndex(it->average_, avg_idx);
    if (frontier_flag_[toAdr(avg_idx)] == FORCE_DORMANT) {
      it = dormant_frontiers_.erase(it);
    }
    else {
      ++it;
    }
  }
}

void FrontierMap2D::getFrontiers(
    vector<vector<Eigen::Vector2d>>& clusters, vector<Vector2d>& averages)
{
  clusters.clear();
  averages.clear();

  // Extract cluster data from all active frontiers
  for (auto frontier : frontiers_) {
    clusters.push_back(frontier.cells_);
    averages.push_back(frontier.average_);
  }
}

void FrontierMap2D::getDormantFrontiers(
    vector<vector<Eigen::Vector2d>>& clusters, vector<Vector2d>& averages)
{
  clusters.clear();
  averages.clear();
  for (auto frontier : dormant_frontiers_) {
    clusters.push_back(frontier.cells_);
    averages.push_back(frontier.average_);
  }
}

void FrontierMap2D::getFrontierBoxes(vector<pair<Eigen::Vector2d, Eigen::Vector2d>>& boxes)
{
  boxes.clear();
  for (auto frontier : frontiers_) {
    Vector2d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector2d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

bool FrontierMap2D::isSatisfyFrontier(const Eigen::Vector2i& idx)
{
  if (sdf_map_->getInflateOccupancy(idx))
    return false;
  // if (sdf_map_->isInMap(idx) && knownFree(idx) && isNeighborUnknown(idx))
  if (sdf_map_->isInMap(idx) && knownUnknown(idx) && isNeighborFree(idx))
    return true;
  return false;
}

}  // namespace apexnav_planner