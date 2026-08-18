#ifndef _EXPLORATION_FSM_POLICY_H_
#define _EXPLORATION_FSM_POLICY_H_

namespace apexnav_planner {

inline bool isFailedForwardAction(
    int last_action, int forward_action, double planar_displacement, double stuck_distance)
{
  return last_action == forward_action && planar_displacement < stuck_distance;
}

inline const char* stateName(int state)
{
  switch (state) {
    case 0:
      return "INIT";
    case 1:
      return "WAIT_TRIGGER";
    case 2:
      return "PLAN_ACTION";
    case 3:
      return "WAIT_ACTION_FINISH";
    case 4:
      return "PUB_ACTION";
    case 5:
      return "FINISH";
    case 6:
      return "FINISH_FAILURE";
    default:
      return "UNKNOWN";
  }
}

}  // namespace apexnav_planner

#endif
