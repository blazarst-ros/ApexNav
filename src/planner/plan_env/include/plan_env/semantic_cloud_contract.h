#ifndef PLAN_ENV_SEMANTIC_CLOUD_CONTRACT_H_
#define PLAN_ENV_SEMANTIC_CLOUD_CONTRACT_H_

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

namespace apexnav_planner {

inline bool isFiniteXYZ32PointCloud(const sensor_msgs::PointCloud2& cloud)
{
  // A detector mask may legitimately have no depth-supported points. ROS
  // represents that as width=0, height=1 with a complete XYZ field layout.
  if (cloud.height == 0 || cloud.point_step == 0 || cloud.is_bigendian)
    return false;

  const size_t width = static_cast<size_t>(cloud.width);
  const size_t height = static_cast<size_t>(cloud.height);
  const size_t point_step = static_cast<size_t>(cloud.point_step);
  const size_t row_step = static_cast<size_t>(cloud.row_step);
  if (width > std::numeric_limits<size_t>::max() / point_step ||
      row_step < width * point_step ||
      (row_step != 0 && height > std::numeric_limits<size_t>::max() / row_step) ||
      cloud.data.size() != height * row_step)
    return false;

  size_t offsets[3] = {0, 0, 0};
  bool found[3] = {false, false, false};
  for (const auto& field : cloud.fields) {
    int coordinate = -1;
    if (field.name == "x")
      coordinate = 0;
    else if (field.name == "y")
      coordinate = 1;
    else if (field.name == "z")
      coordinate = 2;
    if (coordinate < 0)
      continue;
    if (found[coordinate] || field.datatype != sensor_msgs::PointField::FLOAT32 ||
        field.count != 1 || static_cast<size_t>(field.offset) + sizeof(float) > point_step)
      return false;
    offsets[coordinate] = static_cast<size_t>(field.offset);
    found[coordinate] = true;
  }
  if (!found[0] || !found[1] || !found[2])
    return false;

  for (size_t row = 0; row < height; ++row) {
    const size_t row_offset = row * row_step;
    for (size_t column = 0; column < width; ++column) {
      const size_t point_offset = row_offset + column * point_step;
      for (size_t coordinate = 0; coordinate < 3; ++coordinate) {
        float value = 0.0f;
        std::memcpy(&value, cloud.data.data() + point_offset + offsets[coordinate],
            sizeof(value));
        if (!std::isfinite(value))
          return false;
      }
    }
  }
  return true;
}

}  // namespace apexnav_planner

#endif  // PLAN_ENV_SEMANTIC_CLOUD_CONTRACT_H_
