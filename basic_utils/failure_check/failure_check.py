from params import FINAL_RESULT, EXPL_RESULT, ROS_STATE


def should_end_multiagent_episode(
    ros_states,
    step_counts,
    max_episode_steps,
    reach_stop_queued,
    reach_claim_stop_executed,
):
    """Return whether a cooperative multi-agent episode may terminate.

    A planner failure is local to one agent.  It becomes a team failure only
    after every remaining agent has either entered FINISH_FAILURE or exhausted
    Habitat's hard step budget.  A reach claim has its own stronger handshake:
    termination waits until the claiming agent's Habitat STOP was executed.
    """
    return get_multiagent_termination_reason(
        ros_states,
        step_counts,
        max_episode_steps,
        reach_stop_queued,
        reach_claim_stop_executed,
    ) is not None


def get_multiagent_termination_reason(
    ros_states,
    step_counts,
    max_episode_steps,
    reach_stop_queued,
    reach_claim_stop_executed,
):
    """Return the team-level exit cause, or ``None`` while work remains."""
    if reach_stop_queued:
        return "reach_claim" if reach_claim_stop_executed else None

    if len(ros_states) != len(step_counts):
        raise ValueError("ros_states and step_counts must describe the same agents")
    if not ros_states:
        return None

    if not all(
        state == ROS_STATE.FINISH_FAILURE or steps >= max_episode_steps
        for state, steps in zip(ros_states, step_counts)
    ):
        return None

    if all(state == ROS_STATE.FINISH_FAILURE for state in ros_states):
        return "all_failed"
    return "step_limit"


def get_reach_claim_outcome(
    final_state, claim_agent_idx, claim_distance_to_goal, success_distance, stop_executed
):
    """Classify a planner reach claim with Habitat's native STOP semantics.

    A reach claim is only a candidate-object assertion.  It becomes success
    only when the claiming agent actually executes STOP strictly inside the
    Habitat success radius; otherwise it is a false positive.
    """
    if final_state != FINAL_RESULT.REACH_OBJECT or claim_agent_idx is None:
        return None
    if not stop_executed:
        return "pending stop"
    if claim_distance_to_goal < success_distance:
        return "success"
    return "false positive"


def is_on_same_floor(height, ref_floor_height=None, ceiling_height=2.0, episode=None):
    """
    Check if a position is on the same floor as a reference height

    Args:
        height (float): Height to check
        ref_floor_height (float, optional): Reference floor height. Uses episode start if None
        ceiling_height (float): Height of ceiling (default: 2.0m)
        episode: Episode object containing start position

    Returns:
        int: 1 if on same floor, 0 otherwise
    """
    if ref_floor_height is None:
        ref_floor_height = episode.start_position[1]
    if ref_floor_height <= height < ref_floor_height + ceiling_height:
        return 1
    else:
        return 0


def check_failure(
    episode, final_state, expl_result, count_steps, max_step, pass_object, near_object
):
    """
    Analyze and categorize navigation failure types

    This function provides comprehensive failure analysis by examining agent behavior,
    exploration results, and spatial relationships with target objects to determine
    the specific type of navigation failure that occurred.

    Args:
        episode: Navigation episode containing goals and environment info
        final_state: Final state when navigation ended
        expl_result: Exploration algorithm result
        count_steps (int): Number of steps taken
        max_step (int): Maximum allowed steps
        pass_object (bool): Whether agent passed near target object
        near_object (bool): Whether agent ended near target object

    Returns:
        str: Categorized failure type for analysis

    Failure Categories:
        - "infeasible": Target not reachable (different floor)
        - "false positive": Agent thinks it found object but wrong location
        - "false negative": Agent passed object but didn't detect it
        - "no frontier": No more explorable areas
        - "stepout feasible": Reached step limit but target was reachable
        - "stucking": Agent got stuck during navigation
    """

    if count_steps == max_step:
        step_out = True
    else:
        step_out = False

    failure = "unkonwn failure"
    is_feasible = 0
    for goal in episode.goals:
        height = goal.position[1]
        is_feasible += is_on_same_floor(height=height, episode=episode)

    # Agent and object are not on the same floor
    if not is_feasible:
        failure = "infeasible"

    else:
        # Active stop
        if not step_out and final_state != FINAL_RESULT.STUCKING:
            # No more frontiers
            if (
                final_state == FINAL_RESULT.NO_FRONTIER
                or expl_result == EXPL_RESULT.SEARCH_EXTREME
            ):
                if pass_object:
                    failure = "[no frontier] false negative"
                else:
                    failure = "no frontier"
            # Think found object but found wrong one
            elif final_state == FINAL_RESULT.REACH_OBJECT and not near_object:
                failure = "false positive"
            else:
                failure = "unknown failure (active stop)"
        # Passive stop
        else:
            # Happened to stop near object but didn't find object
            if near_object:
                failure = "stepout true negative"
            # Didn't stop near object
            else:
                if final_state == FINAL_RESULT.STUCKING:
                    if pass_object:
                        failure = "[stucking] false negative"
                    else:
                        failure = "stucking"
                # Previously passed by object
                else:
                    if pass_object:
                        failure = "[stepout] false negative"
                    # Didn't pass by object but on the same floor
                    else:
                        failure = "stepout feasible"

    return failure
