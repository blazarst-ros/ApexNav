#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm_traj.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_datatypes.h>

namespace apexnav_planner {

void ExplorationFSMReal::init(ros::NodeHandle& nh)  // Initialize the Exploration FSM
{
  nh_ = nh;
  fp_.reset(new FSMParam);  // 创建FSMParam类的实例，并将其托管给智能指针fp_，初始化「FSM 参数容器」
  fd_.reset(new FSMData);  // 创建FSMData类的实例，并托管给智能指针fd_，初始化「FSM 运行时数据容器」

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);  // 创建「探索管理器」ExplorationManager的实例，托管给智能指针expl_manager_，实例化
                                                // FSM 的核心依赖模块
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(
      nh));  // 创建「规划可视化工具」PlanningVisualization的实例，并完成初始化，为后续探索过程的可视化渲染做准备
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstantsReal::VIS_SCALE_FACTOR;

  state_ = RealFSM::State::INIT;  // 初始化有限状态机的状态为 INIT，表示 FSM
                                  // 刚刚启动，等待必要的前提条件满足后才能进入下一状态

  // Load real-world specific parameters(从参数服务器加载参数)
  nh.param("fsm/replan_time", fp_->replan_time_, 0.2);
  nh.param("fsm/replan_traj_end_threshold", fp_->replan_traj_end_threshold_, 1.0);
  nh.param("fsm/replan_frontier_change_delay", fp_->replan_frontier_change_delay_, 0.5);
  nh.param("fsm/replan_timeout", fp_->replan_timeout_, 2.0);

  /* ROS Timer
   * 创建周期性定时器，每隔指定的时间间隔，就自动调用一次传入的回调函数，直到节点关闭或手动停止定时器*/

  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstantsReal::EXEC_TIMER_DURATION), &ExplorationFSMReal::FSMCallback, this);
  // FSM 主循环定时器
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstantsReal::FRONTIER_TIMER_DURATION),
      &ExplorationFSMReal::frontierCallback, this);
  // 前沿更新定时器
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &ExplorationFSMReal::safetyCallback, this);
  // 安全检查定时器

  /* ROS Subscriber */
  trigger_sub_ = nh.subscribe(
      "/move_base_simple/goal", 10, &ExplorationFSMReal::triggerCallback, this);  // 探索触发指令

  goal_sub_ =
      nh.subscribe("/initialpose", 10, &ExplorationFSMReal::goalCallback, this);  // 目标位姿指令
  odom_sub_ = nh.subscribe(
      "/odom_world", 10, &ExplorationFSMReal::odometryCallback, this);  // 机器人里程计数据
  confidence_threshold_sub_ = nh.subscribe("/detector/confidence_threshold", 10,
      &ExplorationFSMReal::confidenceThresholdCallback, this);  // 障碍物置信度阈值更新安全检测参数

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>(
      "/ros/state", 10);  // 发布当前 FSM 状态,（INIT/WAIT_TRIGGER 等）
  expl_state_pub_ = nh.advertise<std_msgs::Int32>(
      "/ros/expl_state", 10);  // 发布当前探索状态（未探索 / 探索中 / 完成等）
  expl_result_pub_ = nh.advertise<std_msgs::Int32>(
      "/ros/expl_result", 10);  //	发布探索结果（成功 / 失败 / 无可行前沿等）
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>(
      "/robot", 10);  // 发布机器人位姿可视化标记（供 RViz 显示）

  // Real-world trajectory publishers
  poly_traj_pub_ = nh.advertise<trajectory_manager::PolyTraj>(
      "/planning/trajectory", 10);  // 发布规划好的多项式轨迹（供轨迹执行节点）
  stop_pub_ = nh.advertise<std_msgs::Empty>(
      "/traj_server/stop", 10);  // 发布紧急停止指令（让机器人停止轨迹执行）

  ROS_INFO("[ExplorationFSMReal] Initialization complete.");
}

// Main FSM callback for real-world exploration
void ExplorationFSMReal::FSMCallback(
    const ros::TimerEvent& e)  // 状态机的主循环入口（周期性定时器中就进入）
{
  exec_timer_.stop();  // 避免本轮逻辑重复，避免本轮逻辑未执行完就被重复触发（防止竞态）

  // Publish current state
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = static_cast<int>(state_);
  ros_state_pub_.publish(ros_state_msg);

  switch (state_) {
    case RealFSM::State::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        ROS_WARN_THROTTLE(1.0, "[Real] No odom || No target confidence threshold.");
        exec_timer_.start();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(RealFSM::State::WAIT_TRIGGER, "FSM");
      break;
    }

    case RealFSM::State::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "[Real] Waiting for trigger...");
      break;
    }

    case RealFSM::State::FINISH: {
      fd_->static_state_ = true;
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
      }
      ROS_WARN_THROTTLE(1.0, "[Real] Finish exploration!");
      break;
    }

    case RealFSM::State::PLAN_TRAJ: {
      // Plan trajectory based on current state
      if (fd_->static_state_) {
        // Robot is static, use current odometry
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
      }
      else {
        // Robot is moving, predict future state for smooth replanning（保证轨迹平滑性）

        LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;//指向当前正在执行的「旧局部轨迹信息
        double t_plan = (ros::Time::now() - info->start_time).toSec() + fp_->replan_time_;//旧轨迹已执行时长 + 预定义预测间隔==希望获取的未来极短时刻
        t_plan = min(t_plan, info->duration);//
        

        //利用多项式轨迹的「时间查询特性」，直接获取t_plan时刻的位置、速度、加速度，无需复杂计算
        Eigen::Vector3d cur_pos = info->traj.getPos(t_plan);
        Eigen::Vector3d cur_vel = info->traj.getVel(t_plan);
        Eigen::Vector3d cur_acc = info->traj.getAcc(t_plan);

        double cur_yaw = atan2(cur_vel(1), cur_vel(0));

        // 计算预测时间点的「航向角速度（omega）」
        Eigen::Matrix2d B_h;
        B_h << 0, -1.0, 1.0, 0;//2D旋转矩阵
        Eigen::Vector2d cur_vel_2d = cur_vel.head(2);
        Eigen::Vector2d cur_acc_2d = cur_acc.head(2);
        double norm_vel = cur_vel_2d.norm();
        double help1 = 1.0 / (norm_vel * norm_vel + 1e-2);
        double omega = help1 * cur_acc_2d.transpose() * B_h * cur_vel_2d;//提取垂直加速度分量进行点积，v*OMEGA
        
        //将预测结果存入 FSM 运行时数据容器，作为新轨迹的起始状态
        fd_->start_pt_ = cur_pos;
        fd_->start_vel_ = cur_vel;
        fd_->start_yaw_(0) = cur_yaw;
        fd_->start_yaw_(1) = omega;
      }

      TrajPlannerResult res = callTrajectoryPlanner();  
      /*调用轨迹规划器，执行核心规划逻辑
      以你之前预测得到的「未来平滑起始状态」（位置、速度、航向角等）作为输入，
      结合当前的环境信息（如探索前沿、障碍物），
      执行完整的多项式轨迹规划逻辑，
      最终返回一个「规划结果状态」（成功 / 失败 / 任务完成），
      供后续 FSM 进行状态流转决策*/ 


      if (res == TrajPlannerResult::FAILED) {  // 根据规划结果，进行状态流转
        ROS_WARN("[Real] Plan trajectory failed");
        fd_->static_state_ = true;
      }
      else if (res == TrajPlannerResult::SUCCESS) {
        transitState(RealFSM::State::EXEC_TRAJ, "FSM");
      }
      else {  // TrajPlannerResult::MISSION_COMPLETE
        transitState(RealFSM::State::FINISH, "FSM");
      }

      visualize();
      break;
    }

    case RealFSM::State::EXEC_TRAJ: {
      // Publish trajectory and transition to execution monitoring
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {//等到轨迹生效时间到达后，才发布轨迹（否则执行预测瞬时轨迹）
        trajectory_manager::PolyTraj poly_msg;
        polyTraj2ROSMsg(fd_->newest_traj_, poly_msg);
        poly_traj_pub_.publish(poly_msg);
        fd_->static_state_ = false;
        transitState(RealFSM::State::REPLAN, "FSM");
      }
      break;
    }

    case RealFSM::State::REPLAN: {
      // Monitor trajectory execution and decide when to replan
      LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
      double t_cur = (ros::Time::now() - info->start_time).toSec();//该轨迹执行了多久
      double time_to_end = info->duration - t_cur;//当前轨迹的剩余时长

      // Replan if trajectory is almost finished （almost--提前规划）
      if (time_to_end < fp_->replan_traj_end_threshold_) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: traj fully executed");
        exec_timer_.start();
        return;
      }

      // Replan if frontier changed during exploration（不是等轨迹完全执行完毕再规划，而是 “提前预判”）
      if (t_cur > fp_->replan_frontier_change_delay_ &&
          fd_->final_result_ == FINAL_RESULT::EXPLORE &&
          expl_manager_->frontier_map2d_->isAnyFrontierChanged()) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: frontier changed");
        exec_timer_.start();
        return;
      }

      // Replan if trajectory timeout当前轨迹执行超时（容错兜底条件）
      if (t_cur > fp_->replan_timeout_) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: time out");
        exec_timer_.start();
        return;
      }
      break;
    }
  }

  exec_timer_.start();
}

TrajPlannerResult ExplorationFSMReal::callTrajectoryPlanner() 
 /*轨迹规划控制器（用于预测后规划轨迹） 作为上层 FSM 与下层轨迹规划算法（GCopter）之间的「核心桥梁」，
 它完成了「从探索目标提取到轨迹生成」的完整流程 */ 
{
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);//初始化时序,一定时间后开始
  updateFrontierAndObject();

  // Call exploration manager to find next best point
  int expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));



  // Determine final result based on exploration result(确定任务状态,成功/失败/无前沿)
  if (expl_res == EXPL_RESULT::EXPLORATION)
    fd_->final_result_ = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    fd_->final_result_ = FINAL_RESULT::NO_FRONTIER;
  else
    fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;


  // Publish exploration result
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = fd_->final_result_;
  expl_result_pub_.publish(expl_result_msg);

  if (fd_->final_result_ == FINAL_RESULT::NO_FRONTIER) {
    ROS_WARN("[Real] No (passable) frontier");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Select local target from global path
  Eigen::Vector2d goal_pos = expl_manager_->ed_->next_pos_;
  double goal_yaw = 0.0;
  auto path = expl_manager_->ed_->next_best_path_;
  selectLocalTarget(fd_->start_pt_.head(2), path, 4.0, goal_pos, goal_yaw);
  //只规划“当前位置到4米内”的局部目标点
  //在此选定了局部目标点goal_pos和goal_yaw等



  // Check if reached object
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (fd_->start_pt_.head(2) - goal_pos).norm() < 0.25) {
    ROS_ERROR("[Real] Reach the object successfully!");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Prepare state for trajectory planning
  Eigen::VectorXd goal_state(5), current_state(5);// 定义5维的起始/目标状态（GCopter算法要求的输入格式）
  Eigen::Vector3d current_control(0.0, 0.0, 0.0); // 初始控制量（无额外约束）
  double start_vel = Eigen::Vector2d(fd_->start_vel_(0), fd_->start_vel_(1)).norm();// 计算起始速度的大小（只取平面速度，忽略z轴）
  current_state << fd_->start_pt_(0), fd_->start_pt_(1), fd_->start_yaw_(0), 0.0, start_vel;// 填充起始状态：x坐标、y坐标、航向角、航向角速度（设0）、速度大小
  goal_state << goal_pos(0), goal_pos(1), goal_yaw, 0.0, 0.0;
  // 填充目标状态：x坐标、y坐标、目标航向角、航向角速度（设0）、目标速度（设0，到点就停）
  
  
  // Plan trajectory using GCopter（真正的轨迹生成器）
  bool traj_res = expl_manager_->planTrajectory(current_state, goal_state, current_control);
  if (traj_res) {
    auto info = &expl_manager_->gcopter_->local_trajectory_;
    info->start_time = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;
    fd_->newest_traj_ = expl_manager_->gcopter_->local_trajectory_;
    return TrajPlannerResult::SUCCESS;
  }

  return TrajPlannerResult::FAILED;
}

void ExplorationFSMReal::polyTraj2ROSMsg(  
    const LocalTrajectory& local_traj, trajectory_manager::PolyTraj& poly_msg)// 将局部轨迹（LocalTrajectory）转换为 ROS 消息格式发布；
{
  auto data = &local_traj;
  Eigen::VectorXd durs = data->traj.getDurations();
  int piece_num = data->traj.getPieceNum();

  poly_msg.drone_id = 0;// 无人机/机器人ID，单机器人场景设为0即可,d多机器人id需要更改
  poly_msg.traj_id = data->traj_id;// 轨迹ID，用于区分不同轨迹（防重复执行）
  poly_msg.start_time = data->start_time;// 轨迹的起始执行时间（ROS时间戳）
  poly_msg.order = 7;// 多项式轨迹的阶数（7阶多项式）
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);

  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);//把第 i 段轨迹的时长填入 ROS 消息的时长数组。

    auto cMat = data->traj.operator[](i).getCoeffMat();//获取第i段轨迹的系数矩阵（3行8列：x/y/z轴，各8个系数）
    int i8 = i * 8;
    for (int j = 0; j < 8; j++) {
      poly_msg.coef_x[i8 + j] = cMat(0, j); // i8=i*8,确保系数填充位置正确
      poly_msg.coef_y[i8 + j] = cMat(1, j);
      poly_msg.coef_z[i8 + j] = cMat(2, j);
    }
  }
}

void ExplorationFSMReal::selectLocalTarget(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path, const double& local_distance,
    Eigen::Vector2d& target_pos, double& target_yaw)  
// 局部目标选择（被调用的GCopter）为轨迹规划提供一个 “靠谱” 的局部目标，避免规划长距离无效轨迹
//（此处完全没有直接或间接引入「语义得分」相关的逻辑）
{
  
  // First, try to find a collision-free target from the end of path（反向找无碰撞初始目标）
  for (int i = path.size() - 2; i >= 0; i--) {
    target_yaw = atan2(path.back()(1) - path[i](1), path.back()(0) - path[i](0));
    if (!expl_manager_->kinoastar_->isCollisionPosYaw(path[i], target_yaw)) {
      target_pos = path[i];
      break;
    }
  }

  // Find closest path point to current position（找当前位置最近的路径点）
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select local target within local_distance（选指定距离内的局部目标）
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      target_yaw = atan2(path[i](1) - path[i - 1](1), path[i](0) - path[i - 1](0));
      break;
    }
  }

  // Gradient-based safety adjustment（沿SDF梯度调整到安全位置）
  double step_size = 0.05;
  double tolerance = 1e-3;
  int max_iterations = 30;

  for (int i = 0; i < max_iterations; ++i) {
    Eigen::Vector2d prev_pos = target_pos;

    // Get gradient from SDF map
    Eigen::Vector2d grad;
    double dist = expl_manager_->sdf_map_->getDistWithGrad(target_pos, grad);

    if (dist > 0.26)
      break;

    // Move along gradient to safer position
    if (grad.norm() > 1e-6) {
      target_pos += step_size * grad.normalized();
    }

    // Check convergence
    if ((target_pos - prev_pos).norm() < tolerance) {
      break;
    }
  }

  // Store selected local target
  expl_manager_->ed_->next_local_pos_ = target_pos;
}

void ExplorationFSMReal::visualize()  // 实现探索过程的可视化,RVIZ查看
{
  auto ed_ptr = expl_manager_->ed_;

  auto vec2dTo3d = [](const std::vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    std::vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Eigen::Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontiers
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontiers
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw objects
  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Eigen::Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  std::vector<Eigen::Vector2d> local_points;
  local_points.push_back(ed_ptr->next_local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Eigen::Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Eigen::Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);
}

void ExplorationFSMReal::clearVisMarker()  // 清可视化标记
{
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  visualization_->drawLines({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSMReal::updateFrontierAndObject()//负责同步更新「探索前沿地图」和「目标物体地图」的最新状态
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;//auto是必须初始化的自动指针，指向「前沿地图」对象
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d sensor_pos = Eigen::Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->odom_yaw_);
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

void ExplorationFSMReal::frontierCallback(const ros::TimerEvent& e)//保证空闲时环境地图始终是最新的
{
  // Update frontiers and visualize in idle states
  if (state_ != RealFSM::State::WAIT_TRIGGER && state_ != RealFSM::State::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

void ExplorationFSMReal::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{/*由 ROS 话题触发的「探索任务启动回调函数」
  —— 仅当机器人处于「等待触发（WAIT_TRIGGER）」状态时，
  接收外部触发指令（比如点击 RViz 的 2D Pose 工具、上位机发送的启动指令），
  将探索任务标记为 “已触发”，
  并触发 FSM 状态从「WAIT_TRIGGER」切换到「PLAN_TRAJ」，正式启动探索轨迹规划流程*/
  if (state_ != RealFSM::State::WAIT_TRIGGER)
    return;

  fd_->trigger_ = true;
  ROS_INFO("[Real] Exploration triggered!");
  transitState(RealFSM::State::PLAN_TRAJ, "triggerCallback");
}

void ExplorationFSMReal::odometryCallback(const nav_msgs::OdometryConstPtr& msg)//获取机器人实时运动状态的核心入口
{/*实时接收机器人的里程计（Odometry）消息，解析出位置、姿态（航向角）、线速度、角速度等核心运动数据，
  存入 FSM 运行时数据容器（fd_），标记 “已获取里程计数据”，并触发机器人可视化标记的发布*/
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  // Extract linear velocity
  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  // Extract angular velocity
  fd_->odom_omega_(0) = msg->twist.twist.angular.x;
  fd_->odom_omega_(1) = msg->twist.twist.angular.y;
  fd_->odom_omega_(2) = msg->twist.twist.angular.z;

  fd_->have_odom_ = true;

  // Publish robot marker for visualization
  publishRobotMarker();
}

void ExplorationFSMReal::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  /*在首次接收阈值指令时，将外部传入的置信度阈值设置到物体地图（object_map2d_）中，
  标记 “已获取置信度阈值” 并打印日志，
  用于过滤物体检测结果（只保留置信度高于该阈值的物体）*/
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
  ROS_INFO("[Real] Confidence threshold set to: %.2f", msg->data);
}

void ExplorationFSMReal::goalCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)//这是一个由 ROS 目标位姿话题触发的「手动目标点轨迹规划回调函数」
{
  /*接收外部指定的二维目标位姿（x/y/ 航向角），仅当目标点与机器人当前位置距离超过 0.2 米时，
  立即调用 GCopter 算法规划从当前位姿到目标位姿的平滑轨迹*/
  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;

  tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control;
  current_state << fd_->odom_pos_(0), fd_->odom_pos_(1), fd_->odom_yaw_, 0.0, fd_->odom_vel_(0);
  goal_state << x, y, yaw, 0.0, 0.0;
  if ((current_state.head(2) - goal_state.head(2)).norm() > 0.2) {
    current_control << 0.0, 0.0, 0.0;
    expl_manager_->planTrajectory(current_state, goal_state, current_control);
    trajectory_manager::PolyTraj poly_msg;
    polyTraj2ROSMsg(expl_manager_->gcopter_->local_trajectory_, poly_msg);
    poly_traj_pub_.publish(poly_msg);
  }
  ROS_INFO("[Real] Received goal pose: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
}

void ExplorationFSMReal::emergencyStop()
{
  fd_->static_state_ = true;
  stop_pub_.publish(std_msgs::Empty());
}

void ExplorationFSMReal::safetyCallback(const ros::TimerEvent& e)  // 安全监控
{
  if (state_ != RealFSM::State::REPLAN)
    return;

  // Check if robot deviates from planned trajectory
  double t_cur = (ros::Time::now() - expl_manager_->gcopter_->local_trajectory_.start_time).toSec();
  t_cur = min(t_cur, expl_manager_->gcopter_->local_trajectory_.duration);
  Eigen::Vector3d cur_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_cur);

  if ((cur_pos.head(2) - fd_->odom_pos_.head(2)).norm() > 0.3) {
    ROS_ERROR("[Real] Odom far from traj (%.2f, %.2f), Stop!!!", cur_pos(0), cur_pos(1));
    emergencyStop();
    transitState(RealFSM::State::PLAN_TRAJ, "Odom Far From Trajectory");
    return;
  }

  // Time-sampled safety check - use inflated map to detect obstacles
  double time_horizon = 2.5;  // Check trajectory for next 2.5 seconds
  double sample_dt = 0.1;     // Sample every 0.1 seconds

  for (double t_check = t_cur;
      t_check <= min(t_cur + time_horizon, expl_manager_->gcopter_->local_trajectory_.duration);
      t_check += sample_dt) {
    Eigen::Vector3d check_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_check);
    Eigen::Vector2d check_pos_2d = check_pos.head(2);

    // Skip positions too close to origin
    if ((check_pos_2d - Eigen::Vector2d(0.0, 0.0)).norm() < 1.5)
      continue;

    if (expl_manager_->sdf_map_->getInflateOccupancy(check_pos_2d)) {
      ROS_ERROR("[Real] Safety Stop!!! Obstacle detected (%.2f, %.2f) at time %.2f",
          check_pos_2d(0), check_pos_2d(1), t_check);
      emergencyStop();
      transitState(RealFSM::State::PLAN_TRAJ, "Trajectory Safety Stop");
      break;
    }
  }
}

void ExplorationFSMReal::publishRobotMarker()
{
  const double robot_height = FSMConstantsReal::ROBOT_HEIGHT;
  const double robot_radius = FSMConstantsReal::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  robot_marker.scale.x = robot_radius * 2;
  robot_marker.scale.y = robot_radius * 2;
  robot_marker.scale.z = robot_height;

  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  arrow_marker.scale.x = robot_radius + 0.13;
  arrow_marker.scale.y = 0.08;
  arrow_marker.scale.z = 0.08;

  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSMReal::transitState(RealFSM::State new_state, std::string pos_call)
{
  std::string state_str[] = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "EXEC_TRAJ", "REPLAN",
    "FINISH" };
  ROS_INFO("[Real FSM]: %s -> from %s to %s", pos_call.c_str(),
      state_str[static_cast<int>(state_)].c_str(), state_str[static_cast<int>(new_state)].c_str());
  state_ = new_state;
}

}  // namespace apexnav_planner
