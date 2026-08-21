#ifndef APEXNAV_TRAJECTORY_TRACKING_POLICY_H_
#define APEXNAV_TRAJECTORY_TRACKING_POLICY_H_

#include <algorithm>
#include <cmath>

namespace trajectory_manager {

inline double progressRate(double tracking_error, double full_rate_error,
    double freeze_error, double minimum_rate)
{
  if (!std::isfinite(tracking_error) || !std::isfinite(full_rate_error) ||
      !std::isfinite(freeze_error) || !std::isfinite(minimum_rate))
    return 0.0;
  if (tracking_error <= full_rate_error)
    return 1.0;
  if (tracking_error >= freeze_error)
    return 0.0;
  const double span = std::max(1e-6, freeze_error - full_rate_error);
  const double alpha = (tracking_error - full_rate_error) / span;
  return std::max(0.0, minimum_rate + (1.0 - minimum_rate) * (1.0 - alpha));
}

}  // namespace trajectory_manager

#endif
