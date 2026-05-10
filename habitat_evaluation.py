"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Supports both single-agent and multi-agent configurations.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import quaternion
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from llm.answer_reader.semantic_prior_ros import compute_mask_scales, publish_semantic_priors_to_ros
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES, FINAL_RESULT
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def ros_action_callback(msg):
    """Callback for receiving actions from ROS planner.

    Multi-agent encoding: action = agent_idx * 100 + action_code
    Agent 0 sending MOVE_FORWARD (1):  raw = 0*100+1 = 1
    Agent 1 sending TURN_LEFT (2):     raw = 1*100+2 = 102
    Single-agent: msg.data is the raw action code (backward compatible).
    """
    global global_action
    global_action = msg.data


def _make_agent_action_callback(agent_idx: int, actions_dict: dict):
    """Create a per-agent action callback that stores actions separately.

    This avoids the race condition where multiple agents' actions arrive
    simultaneously and overwrite each other in a shared global variable.
    """
    def callback(msg):
        actions_dict[agent_idx] = msg.data
    return callback


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state, mission_reached_object
    if msg.data == FINAL_RESULT.REACH_OBJECT:
        # Treat a planner goal claim as an episode-level stop request.
        # Evaluation still checks Habitat distance at episode end, so a wrong
        # goal claim can be classified as "false positive" like single-agent.
        mission_reached_object = True
        final_state = FINAL_RESULT.REACH_OBJECT
    elif not mission_reached_object:
        final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _get_agent_action_index(agent_idx: int) -> str:
    """Generate the ROS topic suffix for a specific agent's action input."""
    return f"/habitat/plan_action_agent_{agent_idx}"


def _parse_multi_agent_action(raw_action: int):
    """Parse raw action data into (agent_idx, action_code).

    Encoding: action = agent_idx * 100 + action_code

    Backward compatible: if raw < 10, it's single-agent (agent_idx=0).
    """
    if raw_action < 10:
        return 0, raw_action
    agent_idx = raw_action // 100
    action_code = raw_action % 100
    return agent_idx, action_code


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2, or mp3d (default: hm3dv2)",
    )
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def _is_multi_agent(cfg: DictConfig) -> bool:
    """Check if the configuration is for multi-agent mode."""
    return cfg.get("num_agents", 1) > 1


def _setup_multi_agent_env(env, cfg: DictConfig):
    """Assign per-agent start positions.

    Strategy: agent_0 uses episode start_position; additional agents are offset
    around the same floor when navigable.
    Habitat uses integer agent_id (index into agents_order), not string names.
    """
    offset = cfg.get("multiagent", {}).get("agent_spawn_offset", 1.0)
    num_agents = cfg.get("num_agents", 1)
    episode = env.current_episode

    start_pos_0 = list(episode.start_position)
    start_rot_0 = list(episode.start_rotation)
    start_positions = [start_pos_0]

    def candidate_offsets(base_offset):
        return [
            (base_offset, 0.0),
            (-base_offset, 0.0),
            (0.0, base_offset),
            (0.0, -base_offset),
            (base_offset, base_offset),
            (base_offset, -base_offset),
            (-base_offset, base_offset),
            (-base_offset, -base_offset),
        ]

    try:
        pathfinder = env.sim.pathfinder
        for agent_id in range(1, num_agents):
            found_pos = None
            for ring in range(1, 4):
                for dx, dz in candidate_offsets(offset * ring):
                    candidate = [
                        start_pos_0[0] + dx,
                        start_pos_0[1],
                        start_pos_0[2] + dz,
                    ]
                    if not pathfinder.is_navigable(candidate):
                        continue
                    if all(
                        np.linalg.norm(np.array(candidate) - np.array(pos)) > offset * 0.5
                        for pos in start_positions
                    ):
                        found_pos = candidate
                        break
                if found_pos is not None:
                    break
            start_positions.append(found_pos if found_pos is not None else start_pos_0.copy())
    except Exception:
        while len(start_positions) < num_agents:
            start_positions.append(start_pos_0.copy())

    # Use get_agent().set_state() with numpy-typed AgentState instead of
    # env.sim.set_agent_state() because the latter passes Python lists to
    # habitat-sim's internal translate(), which crashes for agent_id != 0.
    for agent_id, pos in enumerate(start_positions):
        agent = env.sim.get_agent(agent_id)
        new_state = agent.get_state()
        new_state.position = np.array(pos, dtype=np.float32)
        new_state.rotation = np.array(start_rot_0, dtype=np.float32)
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors=True)


def _build_agent_obs_dict(observations, multi_agent: bool, agent_names: list):
    """Normalize observations to a dict keyed by agent name."""
    if multi_agent:
        return {name: observations[name] for name in agent_names}
    else:
        return {agent_names[0]: observations}


def _multi_agent_step(env, action_dict: dict, agent_names: list):
    """Step the simulation with per-agent actions using Habitat-Sim directly.

    Habitat.Env only steps the default agent via env.step(). For multi-agent,
    we apply each agent's action via the simulator API, then advance physics once.
    """
    import habitat_sim

    for agent_name, action in action_dict.items():
        if action is not None:
            agent_idx = agent_names.index(agent_name)
            env.sim.get_agent(agent_idx).act(action)

    # Advance the physics simulation once (all queued actions execute together)
    env.sim.step_world(1.0 / 60.0)

    # Get observations from all agents' sensors and restructure by agent
    return _get_agent_observations(env, agent_names)


def _compute_agent_gps(env, agent_idx: int):
    """Compute GPS observation for a specific agent (same math as GPSSensor).

    Returns 2D position in the episode's coordinate frame:
    [-agent_z_in_frame, agent_x_in_frame]
    """
    agent_state = env.sim.get_agent_state(agent_idx)
    episode = env.current_episode

    origin = np.array(episode.start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(episode.start_rotation)

    agent_position = np.array(agent_state.position, dtype=np.float32)
    agent_position = quaternion_rotate_vector(
        rotation_world_start.inverse(), agent_position - origin
    )
    # Return 3D [x, y, z] in episode frame — habitat_publisher accesses gps[0..2]
    return agent_position[:3].astype(np.float32)


def _compute_agent_compass(env, agent_idx: int):
    """Compute compass observation for a specific agent (same math as CompassSensor).

    Returns heading in the episode's coordinate frame as a scalar in [-pi, pi].
    """
    agent_state = env.sim.get_agent_state(agent_idx)
    episode = env.current_episode

    rotation_world_agent = agent_state.rotation
    rotation_world_start = quaternion_from_coeff(episode.start_rotation)

    if isinstance(rotation_world_agent, quaternion.quaternion):
        quat = rotation_world_agent.inverse() * rotation_world_start
    else:
        # numpy array → quaternion
        quat = quaternion_from_coeff(np.array(rotation_world_agent, dtype=np.float32))
        quat = quat.inverse() * rotation_world_start

    direction_vector = np.array([0, 0, -1])
    heading_vector = quaternion_rotate_vector(quat, direction_vector)
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array([phi], dtype=np.float32)


def _get_agent_observations(env, agent_names: list):
    """Get per-agent observations as a dict keyed by agent name.

    MultiAgentSim-v0 returns observations with namespaced sensor UUIDs
    (e.g. agent_0_rgb, agent_0_depth, agent_1_rgb, ...).  We remap
    agent-prefixed uuids back to simple keys (rgb, depth, gps, compass)
    within each agent's sub-dict so downstream code can use
    agent_obs["rgb"], agent_obs["depth"], etc. unchanged.

    GPS and compass are task sensors not available from get_sensor_observations(),
    so we compute them per-agent manually.
    """
    num_agents = len(agent_names)
    sim_obs = env.sim.get_sensor_observations(agent_ids=list(range(num_agents)))

    # If sim_obs is keyed by agent_id (multi-agent), flatten first
    if isinstance(sim_obs, dict) and any(isinstance(k, int) for k in sim_obs.keys()):
        merged = {}
        for aid, aobs in sim_obs.items():
            merged.update(aobs)
        observations = env.sim._sensor_suite.get_observations(merged)
    else:
        observations = env.sim._sensor_suite.get_observations(sim_obs)

    result = {name: {} for name in agent_names}
    for key, val in observations.items():
        for agent_name in agent_names:
            prefix = agent_name + "_"
            if key.startswith(prefix):
                result[agent_name][key[len(prefix):]] = val
                break
        else:
            # Non-agent-prefixed keys (e.g. task sensors) — assign to agent_0
            result[agent_names[0]][key] = val

    # Compute GPS and compass per agent (task sensors not in sim observations)
    for agent_idx, agent_name in enumerate(agent_names):
        result[agent_name]["gps"] = _compute_agent_gps(env, agent_idx)
        result[agent_name]["compass"] = _compute_agent_compass(env, agent_idx)

    return result


def _get_agent_distance_to_goal(env, agent_idx: int):
    """Compute geodesic distance from agent to the nearest goal position."""
    try:
        agent_state = env.sim.get_agent_state(agent_idx)
        agent_pos = agent_state.position
        goals = env.current_episode.goals
        if not goals:
            return 999.0
        distances = []
        for goal in goals:
            dist = env.sim.geodesic_distance(agent_pos, goal.position)
            distances.append(dist)
        return min(distances)
    except Exception:
        return 999.0


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result, mission_reached_object

    multi_agent = _is_multi_agent(cfg)
    num_agents = cfg.get("num_agents", 1)
    agent_names = [f"agent_{i}" for i in range(num_agents)]
    multiagent_cfg = cfg.get("multiagent", {})
    termination_policy = multiagent_cfg.get("episode_termination", "cooperative")
    perception_agents_per_step = max(
        1, int(multiagent_cfg.get("perception_agents_per_step", num_agents))
    )
    # Per-agent viewpoint-change interval for VLM requests. This is not a
    # global scheduler-step interval; only actions by the same agent that
    # change its camera viewpoint advance the counter.
    perception_interval_steps = max(
        1, int(multiagent_cfg.get("perception_interval_steps", 1))
    )
    _itm_pubs = {}
    _cld_pubs = {}

    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    mission_reached_object = False
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector
    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    env_num_once = cfg.test_epi_num
    flag_once = env_num_once != -1

    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add measurements
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True, visibility_dist=5.0, fov=79
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    # ── Create environment ──
    # Habitat does not have MultiAgentEnv; use Env with a multi-agent config.
    # The simulator creates all agents, and we interact with them via agent_id.
    env = habitat.Env(cfg)
    print(f"Environment created ({'multi-agent' if multi_agent else 'single-agent'}, "
          f"{num_agents} agent(s))")
    number_of_episodes = env.number_of_episodes

    # Read previous records
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # ── ROS Setup ──
    agent_actions = {}  # Per-agent action storage (keyed by agent_idx)
    if multi_agent:
        ros_pubs = {}
        for agent_name in agent_names:
            ros_pubs[agent_name] = habitat_publisher.ROSPublisher(agent_name)
        for agent_idx in range(num_agents):
            topic = _get_agent_action_index(agent_idx)
            rospy.Subscriber(topic, Int32, _make_agent_action_callback(agent_idx, agent_actions), queue_size=10)
        ros_pub = ros_pubs[agent_names[0]]
    else:
        obj_point_cloud_pub = rospy.Publisher(
            "habitat/object_point_cloud", PointCloud2, queue_size=10
        )
        ros_pub = habitat_publisher.ROSPublisher()
        # Single-agent: subscribe to the default action topic
        rospy.Subscriber(
            _get_agent_action_index(0), Int32, ros_action_callback, queue_size=10
        )
    # ROS state callbacks for multi-agent tracking
    # Match C++ NUM_AGENTS in simulation mode so readiness checks include all planner agents.
    ros_all_states = [ROS_STATE.INIT] * max(num_agents, 3)
    ros_all_state_stamps = [rospy.Time(0)] * max(num_agents, 3)
    ros_all_state_receive_stamps = [0.0] * max(num_agents, 3)

    def ros_all_state_callback(msg):
        # Legacy compatibility only: this topic has no per-agent timestamp or
        # agent_id, so it must not make state look fresh.
        for i, s in enumerate(msg.data):
            if i < len(ros_all_states):
                ros_all_states[i] = s

    def ros_state_agents_callback(msg):
        record_size = 4
        if len(msg.data) % record_size != 0:
            rospy.logwarn_throttle(
                1.0,
                "Ignoring malformed /ros/state_agents message with %d fields",
                len(msg.data),
            )
            return
        for offset in range(0, len(msg.data), record_size):
            stamp_sec, stamp_nsec, agent_id, state = msg.data[offset : offset + record_size]
            if 0 <= agent_id < len(ros_all_states):
                ros_all_states[agent_id] = state
                ros_all_state_stamps[agent_id] = rospy.Time(stamp_sec, stamp_nsec)
                ros_all_state_receive_stamps[agent_id] = time.monotonic()

    def _fresh_ros_state(agent_idx, max_age_sec=1.0):
        receive_stamp = ros_all_state_receive_stamps[agent_idx]
        if receive_stamp <= 0.0:
            return False
        return time.monotonic() - receive_stamp <= max_age_sec

    def _ros_state_is(agent_idx, state, max_age_sec=1.0):
        return ros_all_states[agent_idx] == state and _fresh_ros_state(agent_idx, max_age_sec)

    def _ros_state_in(agent_idx, states, max_age_sec=1.0):
        return ros_all_states[agent_idx] in states and _fresh_ros_state(agent_idx, max_age_sec)

    rospy.Subscriber("/ros/state_all", Int32MultiArray, ros_all_state_callback, queue_size=10)
    rospy.Subscriber("/ros/state_agents", Int32MultiArray, ros_state_agents_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=30)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )

    def _finish_episode_handshake():
        """Reliably notify the planner FSM to reset before the next episode.

        A single non-latched EPISODE_FINISH can be missed while the C++ FSM is
        busy in WAIT_ACTION_FINISH, but flooding this topic queues repeated
        C++ map resets. Send one request, wait for state feedback, then retry
        slowly only if the FSM never acknowledges the reset.
        """
        global global_action
        if multi_agent:
            agent_actions.clear()
        else:
            global_action = None

        for attempt in range(3):
            for i in range(num_agents):
                ros_all_states[i] = -1
                ros_all_state_stamps[i] = rospy.Time(0)
                ros_all_state_receive_stamps[i] = 0.0
            publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)
            wait_begin = rospy.Time.now()
            while (rospy.Time.now() - wait_begin).to_sec() < 3.0:
                if all(
                    _ros_state_in(i, (ROS_STATE.INIT, ROS_STATE.WAIT_TRIGGER), 1.0)
                    for i in range(num_agents)
                ):
                    return
                rospy.sleep(0.05)
            print(
                "Waiting for planner reset acknowledgement after "
                f"EPISODE_FINISH attempt {attempt + 1}/3..."
            )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(number_of_episodes - num_total):
        publish_int32_array(progress_pub, [num_total, number_of_episodes])
        final_state = 0
        expl_result = 0
        mission_reached_object = False

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # ── Per-agent state ──
        agent_states = {}
        for agent_name in agent_names:
            agent_states[agent_name] = {
                "global_action": None,
                "count_steps": 0,
                "camera_pitch": 0.0,
                "pass_object": 0.0,
                "near_object": 0.0,
                "success": 0.0,
                "spl": 0.0,
                "soft_spl": 0.0,
                "distance_to_goal": 999.0,
                "distance_to_goal_reward": 0.0,
                "vis_frames": [],
                "finished": False,
                "viewpoint_steps_since_perception": perception_interval_steps,
            }

        # ── LLM answer (shared across agents) ──
        label = env.current_episode.object_category
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )
        publish_semantic_priors_to_ros(llm_answer_path, label, llm_answer)

        # ── Episode init ──
        observations = env.reset()

        if multi_agent:
            _setup_multi_agent_env(env, cfg)
            observations = _get_agent_observations(env, agent_names)

        # Normalize observations dict
        agent_obs_dict = _build_agent_obs_dict(observations, multi_agent, agent_names)
        for agent_name in agent_names:
            agent_obs_dict[agent_name]["camera_pitch"] = 0.0
            info = {}
            metrics = env.get_metrics()
            if multi_agent and isinstance(metrics, dict):
                info = metrics.get(agent_name, metrics)
            elif not multi_agent:
                info = metrics
            if need_video:
                frame = observations_to_image(agent_obs_dict[agent_name], info)
                if "top_down_map" in info:
                    info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                agent_states[agent_name]["vis_frames"].append(frame)

        # Snapshot of initial observations for the timer callback (updated each step)
        timer_obs = {name: dict(agent_obs_dict[name]) for name in agent_names}

        def _publish_observations_timer(event):
            """Publish observations while waiting for the planner to become trigger-ready."""
            if multi_agent:
                for agent_name in agent_names:
                    obs = timer_obs.get(agent_name, {})
                    ros_pubs[agent_name].habitat_publish_ros_topic(obs)
            else:
                ros_pub.habitat_publish_ros_topic(timer_obs.get(agent_names[0], {}))
            publish_float64(confidence_threshold_pub, fusion_threshold)

        # Readiness timer: keep odom/depth/confidence fresh, but do not trigger
        # planning until every agent has reached WAIT_TRIGGER below.
        trigger_pub_timer = rospy.Timer(
            rospy.Duration(0.25),
            _publish_observations_timer,
        )
        _finish_episode_handshake()

        print(f"Agents are waiting in the environment! Target: [{label}]")
        if multi_agent:
            print(f"  Agents: {', '.join(agent_names)}")
            print(f"  Termination policy: {termination_policy}")

        rate = rospy.Rate(10)
        while True:
            all_init = all(_ros_state_is(i, ROS_STATE.INIT) for i in range(num_agents))
            any_init = any(_ros_state_is(i, ROS_STATE.INIT) for i in range(num_agents))
            all_wait_trigger = all(
                _ros_state_is(i, ROS_STATE.WAIT_TRIGGER) for i in range(num_agents)
            )
            any_wait_action_finish = any(
                _ros_state_is(i, ROS_STATE.WAIT_ACTION_FINISH)
                for i in range(num_agents)
            )

            if any_wait_action_finish:
                states_str = ", ".join(
                    f"{agent_names[i]}: state={ros_all_states[i]}" for i in range(num_agents)
                )
                print(
                    "Planner entered WAIT_ACTION_FINISH before Python trigger; "
                    f"resetting planner episode state. [{states_str}]"
                )
                _finish_episode_handshake()
                rate.sleep()
                continue

            if all_init:
                print("Waiting for ROS to get odometry for all agents...")
            elif not all_wait_trigger:
                states_str = ", ".join(
                    f"{agent_names[i]}: state={ros_all_states[i]}" for i in range(num_agents)
                )
                print(f"Waiting for trigger: [{states_str}]")

            # Publish odometry/depth for all agents so the C++ planner can detect all agents in INIT state
            if multi_agent:
                for agent_name in agent_names:
                    agent_obs = observations.get(agent_name, {})
                    gps = agent_obs.get("gps", np.array([0.0, 0.0, 0.0]))
                    compass = agent_obs.get("compass", np.array([0.0]))[0]
                    pitch = agent_obs.get("camera_pitch", 0.0)
                    ros_pubs[agent_name].publish_depth(rospy.Time.now(), agent_obs.get("depth"))
                    ros_pubs[agent_name].publish_robot_odom(rospy.Time.now(), gps, compass)
                    ros_pubs[agent_name].publish_camera_odom(rospy.Time.now(), gps, compass, pitch)
                    ros_pubs[agent_name].publish_rgb(rospy.Time.now(), agent_obs.get("rgb"))
            else:
                # Use timer_obs which has "camera_pitch" added; raw observations lacks it
                ros_pub.habitat_publish_ros_topic(timer_obs[agent_names[0]])

            if all_wait_trigger:
                break
            rate.sleep()

        print("Agents are ready to go!!!!")

        # Kick the C++ planner out of WAIT_TRIGGER → PLAN_ACTION
        trigger_pub.publish(PoseStamped())
        rospy.sleep(0.1)

        # ── Main episode loop ──
        # Shut down the timer — the main loop publishes odom on every step
        # and the no-action path (line ~750) publishes at 10Hz when idle.
        # Keeping the timer alive causes dual publishing and TF out-of-order warnings.
        trigger_pub_timer.shutdown()
        rate = rospy.Rate(10)
        global_action = None
        perception_cursor = 0

        def _stop_all_agents_for_evaluation():
            print("Goal claim received from one agent; stopping all agents for evaluation.")
            for stop_agent_name in agent_names:
                agent_states[stop_agent_name]["finished"] = True
            agent_actions.clear()

        while not rospy.is_shutdown():
            # ── Termination check ──
            if multi_agent:
                if mission_reached_object:
                    _stop_all_agents_for_evaluation()
                    break

                for agent_idx, agent_name in enumerate(agent_names):
                    if _ros_state_is(agent_idx, ROS_STATE.FINISH):
                        agent_states[agent_name]["finished"] = True

                if termination_policy == "cooperative":
                    any_finished = any(
                        agent_states[a]["finished"] or agent_states[a]["count_steps"] >= max_episode_steps
                        for a in agent_names
                    )
                    if any_finished:
                        break
                else:
                    all_done = all(
                        agent_states[a]["finished"] or agent_states[a]["count_steps"] >= max_episode_steps
                        for a in agent_names
                    )
                    if all_done:
                        break
            else:
                is_feasible = 0
                for goal in env.current_episode.goals:
                    height = goal.position[1]
                    is_feasible += is_on_same_floor(
                        height=height, episode=env.current_episode
                    )
                if not is_feasible:
                    break

            # ── Parse actions ──
            if multi_agent:
                waiting_action_agents = [
                    agent_idx for agent_idx in range(num_agents)
                    if _ros_state_is(agent_idx, ROS_STATE.WAIT_ACTION_FINISH)
                    and not agent_states[agent_names[agent_idx]]["finished"]
                ]
                if waiting_action_agents and not all(
                    agent_idx in agent_actions for agent_idx in waiting_action_agents
                ):
                    for agent_name in agent_names:
                        ast = agent_states[agent_name]
                        if ast["finished"]:
                            continue
                        gps = _compute_agent_gps(env, agent_names.index(agent_name))
                        compass = _compute_agent_compass(env, agent_names.index(agent_name))
                        ros_pubs[agent_name].publish_robot_odom(rospy.Time.now(), gps, compass)
                    rate.sleep()
                    continue

                # Process per-agent actions from separate ROS subscribers
                # agent_idx is already correct from the per-topic subscription callback;
                # the raw_action value is the plain action code (< 10), NOT encoded as agent_idx*100+code
                actionable_agents = waiting_action_agents or sorted(agent_actions.keys())
                for agent_idx in actionable_agents:
                    action_code = agent_actions.get(agent_idx)
                    if action_code is None:
                        continue
                    if agent_idx < num_agents:
                        aname = f"agent_{agent_idx}"
                        if (
                            action_code == ACTION.MOVE_FORWARD
                            and agent_states[aname]["count_steps"] >= max_episode_steps - 1
                        ):
                            action_code = ACTION.STOP
                        agent_states[aname]["global_action"] = action_code
                    agent_actions.pop(agent_idx, None)
                if mission_reached_object:
                    _stop_all_agents_for_evaluation()
                    break
            else:
                if global_action is not None:
                    if (
                        global_action == ACTION.MOVE_FORWARD
                        and agent_states[agent_names[0]]["count_steps"] >= max_episode_steps - 1
                    ):
                        global_action = ACTION.STOP
                    agent_states[agent_names[0]]["global_action"] = global_action
                    global_action = None

            # ── Build action dict / single action ──
            action_dict = {} if multi_agent else None
            single_action = None
            any_agent_acting = False
            movement_action_agents = set()
            viewpoint_action_agents = set()
            single_action_is_movement = False
            single_action_changes_viewpoint = False

            for agent_name in agent_names:
                ast = agent_states[agent_name]
                g_action = ast.pop("global_action", None)
                if g_action is None:
                    continue
                # Skip finished agents — their actions are no longer relevant
                if ast["finished"]:
                    continue

                any_agent_acting = True
                action = None

                if g_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                    movement_action_agents.add(agent_name)
                    viewpoint_action_agents.add(agent_name)
                    if not multi_agent:
                        single_action_is_movement = True
                        single_action_changes_viewpoint = True
                elif g_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                    viewpoint_action_agents.add(agent_name)
                    if not multi_agent:
                        single_action_changes_viewpoint = True
                elif g_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                    viewpoint_action_agents.add(agent_name)
                    if not multi_agent:
                        single_action_changes_viewpoint = True
                elif g_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    ast["camera_pitch"] -= np.pi / 6.0
                    viewpoint_action_agents.add(agent_name)
                    if not multi_agent:
                        single_action_changes_viewpoint = True
                elif g_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    ast["camera_pitch"] += np.pi / 6.0
                    viewpoint_action_agents.add(agent_name)
                    if not multi_agent:
                        single_action_changes_viewpoint = True
                elif g_action == ACTION.STOP:
                    action = HabitatSimActions.stop
                    ast["finished"] = True

                if multi_agent:
                    action_dict[agent_name] = action
                else:
                    single_action = action

            if multi_agent and mission_reached_object:
                _stop_all_agents_for_evaluation()
                break

            if not any_agent_acting:
                # Still publish odom so the C++ planner and RViz stay updated
                for agent_name in agent_names:
                    ast = agent_states[agent_name]
                    if ast["finished"]:
                        continue
                    if multi_agent:
                        gps = _compute_agent_gps(env, agent_names.index(agent_name))
                        compass = _compute_agent_compass(env, agent_names.index(agent_name))
                        ros_pubs[agent_name].publish_robot_odom(rospy.Time.now(), gps, compass)
                    else:
                        gps = observations.get("gps", np.array([0.0, 0.0, 0.0]))
                        compass = observations.get("compass", np.array([0.0]))[0]
                        ros_pub.publish_robot_odom(rospy.Time.now(), gps, compass)
                rate.sleep()
                continue

            # ── Execute step ──
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            if multi_agent:
                active_actions = {
                    k: v for k, v in action_dict.items()
                    if not agent_states[k]["finished"] and v is not None
                }
                if active_actions:
                    observations = _multi_agent_step(env, active_actions, agent_names)
                if mission_reached_object:
                    _stop_all_agents_for_evaluation()
                    break
                # Update task measurements for the default agent (for spl, etc.)
                # Build a flat observation dict from the per-agent split format
                # and pick a non-stop action to avoid incorrectly setting is_stop_called.
                flat_obs = {}
                for aname, aobs in observations.items():
                    for k, v in aobs.items():
                        flat_obs[f"{aname}_{k}"] = v
                # Use the first non-stop action; if all are stop, use move_forward
                # to avoid falsely triggering is_stop_called for unfinished episodes
                non_stop_actions = [
                    v for k, v in active_actions.items()
                    if v != HabitatSimActions.stop
                ]
                measure_action = (
                    non_stop_actions[0] if non_stop_actions
                    else (list(active_actions.values())[0] if active_actions else HabitatSimActions.move_forward)
                )
                env._task.measurements.update_measures(
                    episode=env.current_episode,
                    action={"action": measure_action},
                    task=env._task,
                    observations=flat_obs,
                )
            else:
                observations = env.step(single_action)
                if env.episode_over:
                    break

            # ── Process per-agent results ──
            if multi_agent:
                # Update timer observation snapshot so the timer callback publishes latest data
                for agent_name in agent_names:
                    agent_obs = observations.get(agent_name, {})
                    agent_obs["camera_pitch"] = agent_states[agent_name]["camera_pitch"]
                    timer_obs[agent_name] = dict(agent_obs)

                metrics = env.get_metrics()
                best_agent = None
                best_dist = float("inf")

                for agent_name in agent_names:
                    ast = agent_states[agent_name]
                    if agent_name in movement_action_agents:
                        ast["count_steps"] += 1
                    if agent_name in viewpoint_action_agents:
                        ast["viewpoint_steps_since_perception"] += 1
                    agent_obs = observations.get(agent_name, {})

                    # Compute per-agent distance to goal via simulator
                    agent_idx = agent_names.index(agent_name)
                    dtg = _get_agent_distance_to_goal(env, agent_idx)
                    ast["distance_to_goal"] = min(ast["distance_to_goal"], dtg)

                    if dtg <= success_distance:
                        ast["near_object"] = 1
                        ast["pass_object"] = max(ast["pass_object"], 1)
                        if dtg <= success_distance:
                            ast["success"] = 1
                            ast["finished"] = True

                    # Use task metrics (from default agent) as approximation for spl/soft_spl
                    ast["spl"] = metrics.get("spl", 0.0) if isinstance(metrics, dict) else 0.0
                    ast["soft_spl"] = metrics.get("soft_spl", 0.0) if isinstance(metrics, dict) else 0.0
                    ast["distance_to_goal_reward"] = metrics.get(
                        "distance_to_goal_reward", 0.0
                    ) if isinstance(metrics, dict) else 0.0

                    if dtg < best_dist:
                        best_dist = dtg
                        best_agent = agent_name

                    # Always publish the latest state so the C++ planner does
                    # not plan from stale odometry when perception is throttled.
                    agent_obs["camera_pitch"] = ast["camera_pitch"]
                    ros_pubs[agent_name].habitat_publish_ros_topic(agent_obs)

                perception_agent_names = []
                perception_candidates = [
                    name for name in agent_names
                    if (
                        name in viewpoint_action_agents
                        and not agent_states[name]["finished"]
                        and agent_states[name]["viewpoint_steps_since_perception"]
                        >= perception_interval_steps
                    )
                ]
                if perception_candidates:
                    budget = min(perception_agents_per_step, len(perception_candidates))
                    while len(perception_agent_names) < budget:
                        candidate = agent_names[perception_cursor % num_agents]
                        perception_cursor += 1
                        if (
                            candidate in perception_candidates
                            and candidate not in perception_agent_names
                        ):
                            perception_agent_names.append(candidate)

                for agent_name in perception_agent_names:
                    if mission_reached_object:
                        break
                    ast = agent_states[agent_name]
                    agent_obs = observations.get(agent_name, {})
                    ast["viewpoint_steps_since_perception"] = 0

                    # ITM score
                    img_np = agent_obs.get("rgb", np.zeros((480, 640, 3), dtype=np.uint8))
                    cosine = get_itm_message_cosine(img_np, label, room)
                    itm_score_pub_name = f"/blip2/{agent_name}/cosine_score"
                    if itm_score_pub_name not in _itm_pubs:
                        _itm_pubs[itm_score_pub_name] = rospy.Publisher(itm_score_pub_name, Float64, queue_size=10)
                    _itm_pubs[itm_score_pub_name].publish(Float64(cosine))

                    # Object detection
                    det_img, score_list, object_masks_list, label_list = get_object(
                        label, img_np, detector_cfg, llm_answer
                    )
                    agent_obs["rgb"] = det_img
                    agent_obs["camera_pitch"] = ast["camera_pitch"]
                    ros_pubs[agent_name].habitat_publish_ros_topic(agent_obs)

                    # Point clouds
                    obj_point_cloud_list = get_object_point_cloud(
                        cfg, agent_obs, object_masks_list, agent_name
                    )
                    cld_msg = MultipleMasksWithConfidence()
                    cld_msg.point_clouds = obj_point_cloud_list
                    cld_msg.confidence_scores = score_list
                    cld_msg.label_indices = label_list
                    cld_msg.mask_scales = compute_mask_scales(object_masks_list)
                    cld_pub_name = f"/detector/{agent_name}/clouds_with_scores"
                    if cld_pub_name not in _cld_pubs:
                        _cld_pubs[cld_pub_name] = rospy.Publisher(
                            cld_pub_name, MultipleMasksWithConfidence, queue_size=10
                        )
                    _cld_pubs[cld_pub_name].publish(cld_msg)

                    # Video
                    if need_video:
                        frame = observations_to_image(agent_obs, metrics)
                        if isinstance(metrics, dict) and "top_down_map" in metrics:
                            info_copy = dict(metrics)
                            info_copy.pop("top_down_map")
                            frame = overlay_frame(frame, info_copy)
                        else:
                            frame = overlay_frame(frame, metrics)
                        ast["vis_frames"].append(frame)

                print(f"\n--------------Step: {agent_states[agent_names[0]]['count_steps']}--------------")
                print(f"  Target object: {label}")
                print(f"  Best agent: {best_agent} (dist={best_dist:.3f})")
                publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            else:
                # ── Single-agent processing ──
                ast = agent_states[agent_names[0]]
                if single_action_is_movement:
                    ast["count_steps"] += 1
                if single_action_changes_viewpoint:
                    ast["viewpoint_steps_since_perception"] += 1
                info = env.get_metrics()

                observations["camera_pitch"] = ast["camera_pitch"]
                timer_obs[agent_names[0]] = dict(observations)
                ros_pub.habitat_publish_ros_topic(observations)

                should_run_perception = (
                    single_action_changes_viewpoint
                    and ast["viewpoint_steps_since_perception"] >= perception_interval_steps
                )
                if should_run_perception:
                    ast["viewpoint_steps_since_perception"] = 0
                    cosine = get_itm_message_cosine(observations["rgb"], label, room)
                    print(f"Target related room: {room}")
                    print(f"ITM cosine similarity: {cosine:.3f}")
                    publish_float64(itm_score_pub, cosine)

                    observations["rgb"], score_list, object_masks_list, label_list = get_object(
                        label, observations["rgb"], detector_cfg, llm_answer
                    )

                    observations["camera_pitch"] = ast["camera_pitch"]
                    timer_obs[agent_names[0]] = dict(observations)
                    ros_pub.habitat_publish_ros_topic(observations)

                    cld_msg = MultipleMasksWithConfidence()
                    cld_msg.point_clouds = get_object_point_cloud(
                        cfg, observations, object_masks_list
                    )
                    cld_msg.confidence_scores = score_list
                    cld_msg.label_indices = label_list
                    cld_msg.mask_scales = compute_mask_scales(object_masks_list)
                    cld_with_score_pub.publish(cld_msg)

                ast["distance_to_goal"] = info["distance_to_goal"]
                if ast["distance_to_goal"] <= success_distance and ast["pass_object"] == 0:
                    ast["pass_object"] = 1
                ast["success"] = info["success"]
                ast["spl"] = info["spl"]
                ast["soft_spl"] = info["soft_spl"]
                ast["distance_to_goal_reward"] = info["distance_to_goal_reward"]

                if need_video:
                    frame = observations_to_image(observations, info)
                    info.pop("top_down_map")
                    frame = overlay_frame(frame, info)
                    ast["vis_frames"].append(frame)

                print(f"Finding [{label}]; Action: {single_action};")
                publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)

            rate.sleep()

        # Timer was already shut down before the main loop to avoid dual publishing
        # (causes TF out-of-order warnings and agent stuck issues)

        # ── Episode-end processing ──
        _finish_episode_handshake()

        # Aggregate metrics
        if multi_agent:
            any_success = any(agent_states[a]["success"] == 1 for a in agent_names)
            best_agent = min(agent_names, key=lambda a: agent_states[a]["distance_to_goal"])
            spl = agent_states[best_agent]["spl"]
            soft_spl = agent_states[best_agent]["soft_spl"]
            distance_to_goal = agent_states[best_agent]["distance_to_goal"]
            distance_to_goal_reward = agent_states[best_agent]["distance_to_goal_reward"]
            success = 1 if any_success else 0
            best_ast = agent_states[best_agent]

            print(f"\n------ Episode End ------")
            for agent_name in agent_names:
                ast = agent_states[agent_name]
                print(f"  {agent_name}: steps={ast['count_steps']}, "
                      f"dist={ast['distance_to_goal']:.3f}, success={ast['success']}")
        else:
            ast = agent_states[agent_names[0]]
            spl = ast["spl"]
            soft_spl = ast["soft_spl"]
            distance_to_goal = ast["distance_to_goal"]
            distance_to_goal_reward = ast["distance_to_goal_reward"]
            success = ast["success"]
            best_ast = ast

        near_object = 1 if distance_to_goal <= success_distance else 0

        if success == 1:
            num_success += 1
            result_text = "success"
            best_ast["near_object"] = 1
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                best_ast["count_steps"],
                max_episode_steps,
                best_ast.get("pass_object", 0),
                best_ast.get("near_object", 0),
            )

        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)
        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                best_ast["vis_frames"], img2video_output_path, video_name, fps=6, quality=9
            )

        # Display metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        if multi_agent:
            table1.add_row(["Termination Policy", termination_policy])
            for agent_name in agent_names:
                ast = agent_states[agent_name]
                table1.add_row([f"{agent_name} steps", ast["count_steps"]])
                table1.add_row([f"{agent_name} success", "Yes" if ast["success"] == 1 else "No"])
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        write_record(
            scene_id, episode_id, table1, result_text, label, num_total,
            time_spend, record_file_path,
        )
        write_record(
            scene_id, episode_id, table2, result_text, label, num_total,
            time_spend, continue_path,
        )

        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]
            folder_path = os.path.join(video_output_path, folder)
            result_list[i] = count_files_in_directory(folder_path)

        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        if not multi_agent:
            rospy.sleep(0.1)

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
