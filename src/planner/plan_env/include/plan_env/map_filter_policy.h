#ifndef PLAN_ENV_MAP_FILTER_POLICY_H_
#define PLAN_ENV_MAP_FILTER_POLICY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace apexnav_planner {

struct SelfFilterFootprint {
  double length = 0.70;
  double width = 0.70;
  double tolerance = 1e-6;
};

// Transform the world point into the yaw-aligned body frame before applying
// the rectangular airframe boundary. This deliberately is not a circumcircle:
// an external point near a footprint corner must remain an obstacle return.
inline bool isInsideSelfFootprint(double world_x, double world_y, double body_x,
    double body_y, double yaw, const SelfFilterFootprint& footprint)
{
  if (!std::isfinite(world_x) || !std::isfinite(world_y) || !std::isfinite(body_x) ||
      !std::isfinite(body_y) || !std::isfinite(yaw) || !(footprint.length > 0.0) ||
      !(footprint.width > 0.0) || footprint.tolerance < 0.0)
    return false;
  const double dx = world_x - body_x;
  const double dy = world_y - body_y;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double body_forward = c * dx + s * dy;
  const double body_left = -s * dx + c * dy;
  return std::abs(body_forward) <= 0.5 * footprint.length + footprint.tolerance &&
         std::abs(body_left) <= 0.5 * footprint.width + footprint.tolerance;
}

inline bool isAcceptableMapSourceStamp(double source_stamp_sec, double now_sec,
    double previous_stamp_sec, double max_age_sec, double max_future_sec)
{
  return std::isfinite(source_stamp_sec) && std::isfinite(now_sec) &&
         std::isfinite(previous_stamp_sec) && source_stamp_sec > 0.0 && now_sec >= 0.0 &&
         max_age_sec >= 0.0 && max_future_sec >= 0.0 && source_stamp_sec <= now_sec + max_future_sec &&
         now_sec - source_stamp_sec <= max_age_sec &&
         (previous_stamp_sec <= 0.0 || source_stamp_sec > previous_stamp_sec);
}

inline bool isSourceStampCoherent(
    double first_stamp_sec, double second_stamp_sec, double max_delta_sec)
{
  return std::isfinite(first_stamp_sec) && std::isfinite(second_stamp_sec) &&
         std::isfinite(max_delta_sec) && first_stamp_sec > 0.0 && second_stamp_sec > 0.0 &&
         max_delta_sec >= 0.0 &&
         std::abs(first_stamp_sec - second_stamp_sec) <= max_delta_sec;
}

inline bool isValidSemanticDetectionMetadata(
    int label, double confidence, int semantic_class_count = 5)
{
  return semantic_class_count > 0 && label >= 0 && label < semantic_class_count &&
         std::isfinite(confidence) && confidence >= 0.0 && confidence <= 1.0;
}

inline bool isValidNestedCloudContract(double observation_stamp_sec,
    const std::string& observation_frame, double cloud_stamp_sec,
    const std::string& cloud_frame, double stamp_tolerance_sec = 1e-9)
{
  return !observation_frame.empty() && observation_frame == cloud_frame &&
         std::isfinite(stamp_tolerance_sec) && stamp_tolerance_sec >= 0.0 &&
         isSourceStampCoherent(
             observation_stamp_sec, cloud_stamp_sec, stamp_tolerance_sec);
}

inline bool mayPublishEsdfSnapshot(bool esdf_update_pending, bool source_stamp_pending)
{
  return !esdf_update_pending && !source_stamp_pending;
}

inline double clampedHistoryCutoffSec(double source_stamp_sec, double history_window_sec)
{
  if (!std::isfinite(source_stamp_sec) || !std::isfinite(history_window_sec) ||
      source_stamp_sec <= 0.0 || history_window_sec < 0.0)
    return 0.0;
  return std::max(0.0, source_stamp_sec - history_window_sec);
}

inline bool mappingTransitionRequiresFullReset(bool mapping_enabled)
{
  // HOLD_READY/AUTO are the only enabled phases. A false -> true transition
  // therefore starts a new mission and must never inherit the prior mission's
  // occupancy, semantic objects or value evidence.
  return mapping_enabled;
}

inline double semanticFovConfidence(double relative_angle, double fov_angle)
{
  if (!std::isfinite(relative_angle) || !std::isfinite(fov_angle) ||
      fov_angle <= 0.0 || fov_angle > 2.0 * M_PI)
    return 0.0;
  const double normalized_angle = std::remainder(relative_angle, 2.0 * M_PI);
  const double half_fov = 0.5 * fov_angle;
  if (std::abs(normalized_angle) > half_fov)
    return 0.0;
  const double cosine = std::cos(
      std::abs(normalized_angle) / half_fov * (0.5 * M_PI));
  const double confidence = cosine * cosine;
  return std::isfinite(confidence) ? confidence : 0.0;
}

inline bool tryFuseSemanticValue(double now_confidence, double now_value,
    double last_confidence, double last_value, double* fused_confidence,
    double* fused_value)
{
  if (!fused_confidence || !fused_value || !std::isfinite(now_confidence) ||
      !std::isfinite(now_value) || !std::isfinite(last_confidence) ||
      !std::isfinite(last_value) || now_confidence < 0.0 ||
      last_confidence < 0.0)
    return false;
  const double total_confidence = now_confidence + last_confidence;
  if (!std::isfinite(total_confidence) || total_confidence <= 0.0)
    return false;
  const double next_confidence =
      (now_confidence * now_confidence + last_confidence * last_confidence) /
      total_confidence;
  const double next_value =
      (now_confidence * now_value + last_confidence * last_value) /
      total_confidence;
  if (!std::isfinite(next_confidence) || !std::isfinite(next_value))
    return false;
  *fused_confidence = next_confidence;
  *fused_value = next_value;
  return true;
}

inline bool isFinitePoseAndReasonableQuaternion(double px, double py, double pz, double qx,
    double qy, double qz, double qw, double min_quaternion_norm = 0.95,
    double max_quaternion_norm = 1.05)
{
  if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) || !std::isfinite(qx) ||
      !std::isfinite(qy) || !std::isfinite(qz) || !std::isfinite(qw) ||
      !(min_quaternion_norm > 0.0) || !(max_quaternion_norm >= min_quaternion_norm))
    return false;
  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  return std::isfinite(norm) && norm >= min_quaternion_norm && norm <= max_quaternion_norm;
}

inline bool isValidMapFrameContract(const std::string& map_frame, const std::string& pose_frame,
    const std::string& pose_child_frame, const std::string& depth_frame,
    const std::string& camera_info_frame)
{
  return !map_frame.empty() && !pose_frame.empty() && !pose_child_frame.empty() &&
         !depth_frame.empty() && !camera_info_frame.empty() && pose_frame == map_frame &&
         pose_child_frame == depth_frame && camera_info_frame == depth_frame;
}

struct InflationGridCell {
  int x;
  int y;
};

// Rebuild target cells from every still-occupied source that can reach them.
// Keeping this as a pure policy makes the overlapping-halo invariant testable.
inline void rebuildInflationRegion(const std::vector<unsigned char>& occupied, int width, int height,
    int target_min_x, int target_max_x, int target_min_y, int target_max_y, double radius_cells,
    std::vector<unsigned char>* inflated)
{
  if (width <= 0 || height <= 0 || !std::isfinite(radius_cells) || radius_cells < 0.0 || !inflated ||
      occupied.size() != static_cast<size_t>(width * height) ||
      inflated->size() != occupied.size())
    return;
  target_min_x = std::max(0, target_min_x);
  target_max_x = std::min(width - 1, target_max_x);
  target_min_y = std::max(0, target_min_y);
  target_max_y = std::min(height - 1, target_max_y);
  const int radius_bound = static_cast<int>(std::ceil(radius_cells));
  for (int x = target_min_x; x <= target_max_x; ++x) {
    for (int y = target_min_y; y <= target_max_y; ++y) {
      bool covered = false;
      for (int sx = std::max(0, x - radius_bound); sx <= std::min(width - 1, x + radius_bound) && !covered;
           ++sx) {
        for (int sy = std::max(0, y - radius_bound); sy <= std::min(height - 1, y + radius_bound); ++sy) {
          const int dx = x - sx;
          const int dy = y - sy;
          if (occupied[sx * height + sy] &&
              static_cast<double>(dx * dx + dy * dy) <= radius_cells * radius_cells) {
            covered = true;
            break;
          }
        }
      }
      (*inflated)[x * height + y] = covered ? 1 : 0;
    }
  }
}

using RaycastEpoch = std::uint32_t;
static_assert(sizeof(RaycastEpoch) > 1, "raycast epochs must not wrap at 8-bit cadence");

// Reset every mutable grid buffer as one validated operation. The occupancy
// sentinel must denote UNKNOWN; derived occupancy, ESDF and raycast state are
// cleared so a new /clock epoch cannot inherit any prior-world evidence.
inline bool resetEpochGridBuffers(size_t buffer_size, double unknown_occupancy,
    double default_distance, std::vector<double>* occupancy,
    std::vector<char>* inflated, std::vector<double>* negative_distance,
    std::vector<double>* distance, std::vector<double>* temporary,
    std::vector<char>* virtual_ground, std::vector<short>* hit,
    std::vector<short>* miss, std::vector<short>* hit_and_miss,
    std::vector<RaycastEpoch>* ray_flags)
{
  if (buffer_size == 0 || !std::isfinite(unknown_occupancy) ||
      !std::isfinite(default_distance) || !occupancy || !inflated ||
      !negative_distance || !distance || !temporary || !virtual_ground ||
      !hit || !miss || !hit_and_miss || !ray_flags)
    return false;

  occupancy->assign(buffer_size, unknown_occupancy);
  inflated->assign(buffer_size, 0);
  negative_distance->assign(buffer_size, default_distance);
  distance->assign(buffer_size, default_distance);
  temporary->assign(buffer_size, 0.0);
  virtual_ground->assign(buffer_size, 0);
  hit->assign(buffer_size, 0);
  miss->assign(buffer_size, 0);
  hit_and_miss->assign(buffer_size, 0);
  ray_flags->assign(buffer_size, 0);
  return true;
}

}  // namespace apexnav_planner

#endif  // PLAN_ENV_MAP_FILTER_POLICY_H_
