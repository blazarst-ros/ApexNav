#ifndef APEXNAV_LOCAL_TARGET_SELECTOR_H_
#define APEXNAV_LOCAL_TARGET_SELECTOR_H_

#include <Eigen/Eigen>

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace apexnav_planner {

struct LocalTargetSelection {
  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  double yaw = 0.0;
  double path_distance = 0.0;
  bool valid = false;
};

// Select the farthest contiguous footprint-safe path point inside the local
// horizon. A collision in the path prefix terminates selection: jumping to a
// later free point would ask the local planner to cross the obstacle that made
// the prefix unsafe.
inline LocalTargetSelection selectFootprintSafeLocalTarget(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path, double local_distance, double minimum_progress,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision)
{
  LocalTargetSelection result;
  result.position = current_pos;
  if (path.empty() || local_distance <= 0.0)
    return result;

  int nearest_index = 0;
  double nearest_distance = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(path.size()); ++i) {
    const double distance = (path[i] - current_pos).norm();
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest_index = i;
    }
  }

  auto tangentYaw = [&path](int index) {
    for (int next = index + 1; next < static_cast<int>(path.size()); ++next) {
      const Eigen::Vector2d delta = path[next] - path[index];
      if (delta.norm() > 1e-6)
        return std::atan2(delta.y(), delta.x());
    }
    for (int previous = index - 1; previous >= 0; --previous) {
      const Eigen::Vector2d delta = path[index] - path[previous];
      if (delta.norm() > 1e-6)
        return std::atan2(delta.y(), delta.x());
    }
    return 0.0;
  };

  double path_distance = 0.0;
  Eigen::Vector2d previous = current_pos;
  for (int i = nearest_index; i < static_cast<int>(path.size()); ++i) {
    const double segment = (path[i] - previous).norm();
    previous = path[i];
    path_distance += segment;
    if (segment <= 1e-6 && (path[i] - current_pos).norm() <= 1e-6)
      continue;
    if (path_distance > local_distance + 1e-6)
      break;

    const double yaw = tangentYaw(i);
    if (is_collision(path[i], yaw))
      break;
    if ((path[i] - current_pos).norm() + 1e-6 < minimum_progress)
      continue;

    result.position = path[i];
    result.yaw = yaw;
    result.path_distance = path_distance;
    result.valid = true;
  }
  return result;
}

}  // namespace apexnav_planner

#endif  // APEXNAV_LOCAL_TARGET_SELECTOR_H_
