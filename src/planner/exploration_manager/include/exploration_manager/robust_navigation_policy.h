#ifndef APEXNAV_ROBUST_NAVIGATION_POLICY_H_
#define APEXNAV_ROBUST_NAVIGATION_POLICY_H_

// ROS-free recovery and target-lock rules, kept small enough for deterministic
// unit tests.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace apexnav_planner {

inline bool shouldReuseLockedTarget(bool have_target, double distance_to_target,
    bool target_unsafe, double lock_age, double max_lock_age, int consecutive_failures,
    int max_failures, double reached_distance, double failure_age = 0.0,
    double minimum_failure_hysteresis = 0.0)
{
  if (!std::isfinite(distance_to_target) || !std::isfinite(lock_age) ||
      !std::isfinite(max_lock_age) || !std::isfinite(reached_distance) ||
      !std::isfinite(failure_age) || !std::isfinite(minimum_failure_hysteresis) ||
      lock_age < 0.0 || max_lock_age < 0.0 || reached_distance < 0.0 ||
      failure_age < 0.0 || minimum_failure_hysteresis < 0.0 ||
      consecutive_failures < 0 || max_failures <= 0)
    return false;
  const bool failure_budget_exhausted = consecutive_failures >= max_failures &&
      failure_age >= minimum_failure_hysteresis;
  return have_target && distance_to_target > reached_distance && !target_unsafe &&
         lock_age <= max_lock_age && !failure_budget_exhausted;
}

inline bool trajectoryStartIsContinuous(double start_error, double tolerance)
{
  return start_error <= tolerance;
}

inline bool shouldAcceptPlannerTrigger(bool waiting_for_trigger, bool navigation_enabled)
{
  return waiting_for_trigger && navigation_enabled;
}

inline bool sourceStampIsAcceptable(double now, double stamp, double previous_stamp,
    double max_age, double max_future)
{
  if (!std::isfinite(now) || !std::isfinite(stamp) || !std::isfinite(previous_stamp) ||
      !std::isfinite(max_age) || !std::isfinite(max_future) || stamp <= 0.0 ||
      previous_stamp < 0.0 || max_age <= 0.0 || max_future < 0.0)
    return false;
  const double age = now - stamp;
  return age >= -max_future && age <= max_age &&
      (previous_stamp <= 0.0 || stamp > previous_stamp);
}

inline bool quaternionIsFiniteAndNormalized(
    double x, double y, double z, double w, double norm_tolerance)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w) ||
      !std::isfinite(norm_tolerance) || norm_tolerance < 0.0)
    return false;
  const double squared_norm = x * x + y * y + z * z + w * w;
  return std::isfinite(squared_norm) && squared_norm > 0.0 &&
      std::abs(std::sqrt(squared_norm) - 1.0) <= norm_tolerance;
}

inline bool confidenceThresholdIsValid(double threshold)
{
  return std::isfinite(threshold) && threshold >= 0.0 && threshold <= 1.0;
}

inline bool trajectoryProgressIsValid(double progress, double previous_progress,
    double trajectory_duration, bool have_progress, double first_progress_limit)
{
  if (!std::isfinite(progress) || !std::isfinite(previous_progress) ||
      !std::isfinite(trajectory_duration) || !std::isfinite(first_progress_limit) ||
      progress < 0.0 || previous_progress < 0.0 || trajectory_duration <= 0.0 ||
      first_progress_limit < 0.0 || progress > trajectory_duration + 1e-6)
    return false;
  if (!have_progress)
    return progress <= std::min(trajectory_duration, first_progress_limit) + 1e-6;
  return progress + 1e-6 >= previous_progress;
}

struct FinalTrajectorySampling {
  bool valid = false;
  double time_step = 0.0;
  std::vector<double> times;
};

struct TrajectoryPieceLayout {
  double duration;
  std::vector<double> coefficients;
};

inline bool hasValidTrajectoryPieceLayout(const std::vector<TrajectoryPieceLayout>& pieces)
{
  if (pieces.empty())
    return false;

  double total_duration = 0.0;
  for (const auto& piece : pieces) {
    if (!std::isfinite(piece.duration) || piece.duration <= 0.0 ||
        piece.coefficients.empty())
      return false;
    for (const double coefficient : piece.coefficients)
      if (!std::isfinite(coefficient))
        return false;
    total_duration += piece.duration;
    if (!std::isfinite(total_duration))
      return false;
  }
  return total_duration > 0.0;
}

inline bool hasReachedFinalObjectApproachGoal(double measured_x, double measured_y,
    double final_goal_x, double final_goal_y, double reach_threshold)
{
  return std::isfinite(measured_x) && std::isfinite(measured_y) &&
         std::isfinite(final_goal_x) && std::isfinite(final_goal_y) &&
         std::isfinite(reach_threshold) && reach_threshold >= 0.0 &&
         std::hypot(measured_x - final_goal_x, measured_y - final_goal_y) <= reach_threshold;
}

struct TrajectorySamplePoint {
  double x;
  double y;
  double z;
};

inline bool trajectorySamplePointIsFinite(const TrajectorySamplePoint& point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

inline double trajectorySampleDistance(
    const TrajectorySamplePoint& first, const TrajectorySamplePoint& second)
{
  return std::hypot(std::hypot(first.x - second.x, first.y - second.y), first.z - second.z);
}

struct AdaptiveTrajectorySample {
  std::size_t piece_index;
  double local_time;
  double global_time;
  TrajectorySamplePoint position;
};

struct AdaptiveFinalTrajectorySampling {
  bool valid = false;
  std::vector<AdaptiveTrajectorySample> samples;
};

inline long double binomialCoefficient(std::size_t n, std::size_t k)
{
  if (k > n)
    return 0.0L;
  k = std::min(k, n - k);
  long double result = 1.0L;
  for (std::size_t i = 1; i <= k; ++i) {
    result *= static_cast<long double>(n - k + i);
    result /= static_cast<long double>(i);
  }
  return result;
}

// Return a proof-carrying upper bound for ||dp/dt|| over [0, duration].
// Each inner vector contains one dimension's power-basis coefficients in
// constant-first order. After normalising time to u in [0, 1], the derivative
// of a Bernstein polynomial lies in the convex hull of its derivative control
// points. The largest control-point norm is therefore a conservative bound;
// unlike fixed-time sampling it cannot alias a finite high-order spike.
inline bool conservativeBernsteinDerivativeNormBound(
    const std::vector<std::vector<double>>& power_coefficients,
    double duration, double* derivative_bound)
{
  if (derivative_bound == nullptr || power_coefficients.empty() ||
      !std::isfinite(duration) || duration <= 0.0)
    return false;

  const std::size_t coefficient_count = power_coefficients.front().size();
  constexpr std::size_t kMaxCoefficientCount = 64;
  if (coefficient_count == 0 || coefficient_count > kMaxCoefficientCount)
    return false;
  for (const auto& dimension : power_coefficients) {
    if (dimension.size() != coefficient_count)
      return false;
    for (const double coefficient : dimension)
      if (!std::isfinite(coefficient))
        return false;
  }

  const std::size_t degree = coefficient_count - 1;
  if (degree == 0) {
    *derivative_bound = 0.0;
    return true;
  }

  const long double duration_ld = static_cast<long double>(duration);
  std::vector<std::vector<long double>> normalized_power(
      power_coefficients.size(), std::vector<long double>(coefficient_count, 0.0L));
  for (std::size_t dimension = 0; dimension < power_coefficients.size(); ++dimension) {
    long double duration_power = 1.0L;
    for (std::size_t power = 0; power < coefficient_count; ++power) {
      const long double value =
          static_cast<long double>(power_coefficients[dimension][power]) * duration_power;
      if (!std::isfinite(value))
        return false;
      normalized_power[dimension][power] = value;
      if (power + 1 < coefficient_count) {
        duration_power *= duration_ld;
        if (!std::isfinite(duration_power))
          return false;
      }
    }
  }

  std::vector<std::vector<long double>> bernstein_controls(
      power_coefficients.size(), std::vector<long double>(coefficient_count, 0.0L));
  for (std::size_t dimension = 0; dimension < power_coefficients.size(); ++dimension) {
    for (std::size_t control = 0; control <= degree; ++control) {
      long double value = 0.0L;
      for (std::size_t power = 0; power <= control; ++power) {
        const long double denominator = binomialCoefficient(degree, power);
        const long double weight = binomialCoefficient(control, power) / denominator;
        value += weight * normalized_power[dimension][power];
      }
      if (!std::isfinite(value))
        return false;
      bernstein_controls[dimension][control] = value;
    }
  }

  long double maximum_norm = 0.0L;
  for (std::size_t control = 0; control < degree; ++control) {
    long double squared_norm = 0.0L;
    for (std::size_t dimension = 0; dimension < power_coefficients.size(); ++dimension) {
      const long double derivative = static_cast<long double>(degree) *
          (bernstein_controls[dimension][control + 1] -
              bernstein_controls[dimension][control]) /
          duration_ld;
      if (!std::isfinite(derivative))
        return false;
      squared_norm += derivative * derivative;
      if (!std::isfinite(squared_norm))
        return false;
    }
    maximum_norm = std::max(maximum_norm, std::sqrt(squared_norm));
  }

  // Long-double evaluation followed by an outward-rounded double conversion
  // keeps the returned value conservative in the representation used by the
  // trajectory validator.
  maximum_norm *= 1.0L + 64.0L * std::numeric_limits<long double>::epsilon();
  if (!std::isfinite(maximum_norm) ||
      maximum_norm > static_cast<long double>(std::numeric_limits<double>::max()))
    return false;
  double rounded_bound = static_cast<double>(maximum_norm);
  if (rounded_bound > 0.0)
    rounded_bound = std::nextafter(rounded_bound, std::numeric_limits<double>::infinity());
  if (!std::isfinite(rounded_bound))
    return false;
  *derivative_bound = rounded_bound;
  return true;
}

inline std::vector<long double> multiplyBernsteinPolynomials(
    const std::vector<long double>& first, const std::vector<long double>& second)
{
  if (first.empty() || second.empty())
    return {};
  const std::size_t first_degree = first.size() - 1;
  const std::size_t second_degree = second.size() - 1;
  const std::size_t result_degree = first_degree + second_degree;
  std::vector<long double> result(result_degree + 1, 0.0L);
  for (std::size_t k = 0; k <= result_degree; ++k) {
    const std::size_t begin = k > second_degree ? k - second_degree : 0;
    const std::size_t end = std::min(first_degree, k);
    for (std::size_t i = begin; i <= end; ++i) {
      const std::size_t j = k - i;
      const long double weight = binomialCoefficient(first_degree, i) *
          binomialCoefficient(second_degree, j) /
          binomialCoefficient(result_degree, k);
      result[k] += weight * first[i] * second[j];
    }
  }
  return result;
}

inline bool subdivideBernsteinPolynomialHalf(const std::vector<long double>& coefficients,
    std::vector<long double>* left, std::vector<long double>* right)
{
  if (coefficients.empty() || left == nullptr || right == nullptr)
    return false;
  const std::size_t degree = coefficients.size() - 1;
  std::vector<long double> work = coefficients;
  left->assign(coefficients.size(), 0.0L);
  right->assign(coefficients.size(), 0.0L);
  (*left)[0] = work[0];
  (*right)[degree] = work[degree];
  for (std::size_t level = 1; level <= degree; ++level) {
    for (std::size_t i = 0; i <= degree - level; ++i) {
      work[i] = 0.5L * (work[i] + work[i + 1]);
      if (!std::isfinite(work[i]))
        return false;
    }
    (*left)[level] = work[0];
    (*right)[degree - level] = work[degree - level];
  }
  return true;
}

// Conservative bound for the regularised tangent-yaw rate used by the
// trajectory server: |vx*ay-vy*ax| / (vx^2+vy^2+regularization).
// Bernstein product coefficients bound the numerator and denominator over an
// interval. Fixed-depth de Casteljau subdivision tightens that safe enclosure
// without ever relying on point samples.
inline bool conservativeBernsteinPlanarYawRateBound(
    const std::vector<std::vector<double>>& planar_power_coefficients,
    double duration, double regularization, std::size_t subdivision_depth,
    double* yaw_rate_bound)
{
  if (yaw_rate_bound == nullptr || planar_power_coefficients.size() != 2 ||
      !std::isfinite(duration) || duration <= 0.0 ||
      !std::isfinite(regularization) || regularization <= 0.0 ||
      subdivision_depth > 16)
    return false;
  const std::size_t coefficient_count = planar_power_coefficients.front().size();
  if (coefficient_count == 0 || coefficient_count > 64 ||
      planar_power_coefficients[1].size() != coefficient_count)
    return false;
  for (const auto& dimension : planar_power_coefficients)
    for (const double coefficient : dimension)
      if (!std::isfinite(coefficient))
        return false;

  const std::size_t degree = coefficient_count - 1;
  if (degree < 2) {
    *yaw_rate_bound = 0.0;
    return true;
  }

  const long double duration_ld = static_cast<long double>(duration);
  std::vector<std::vector<long double>> position_controls(
      2, std::vector<long double>(coefficient_count, 0.0L));
  for (std::size_t dimension = 0; dimension < 2; ++dimension) {
    std::vector<long double> normalized_power(coefficient_count, 0.0L);
    long double duration_power = 1.0L;
    for (std::size_t power = 0; power < coefficient_count; ++power) {
      normalized_power[power] =
          static_cast<long double>(planar_power_coefficients[dimension][power]) * duration_power;
      if (!std::isfinite(normalized_power[power]))
        return false;
      if (power + 1 < coefficient_count) {
        duration_power *= duration_ld;
        if (!std::isfinite(duration_power))
          return false;
      }
    }
    for (std::size_t control = 0; control <= degree; ++control) {
      long double value = 0.0L;
      for (std::size_t power = 0; power <= control; ++power) {
        value += binomialCoefficient(control, power) /
            binomialCoefficient(degree, power) * normalized_power[power];
      }
      if (!std::isfinite(value))
        return false;
      position_controls[dimension][control] = value;
    }
  }

  std::vector<std::vector<long double>> velocity_controls(
      2, std::vector<long double>(degree, 0.0L));
  std::vector<std::vector<long double>> acceleration_controls(
      2, std::vector<long double>(degree - 1, 0.0L));
  for (std::size_t dimension = 0; dimension < 2; ++dimension) {
    for (std::size_t control = 0; control < degree; ++control) {
      velocity_controls[dimension][control] = static_cast<long double>(degree) *
          (position_controls[dimension][control + 1] -
              position_controls[dimension][control]) /
          duration_ld;
      if (!std::isfinite(velocity_controls[dimension][control]))
        return false;
    }
    for (std::size_t control = 0; control + 1 < degree; ++control) {
      acceleration_controls[dimension][control] =
          static_cast<long double>(degree - 1) *
          (velocity_controls[dimension][control + 1] -
              velocity_controls[dimension][control]) /
          duration_ld;
      if (!std::isfinite(acceleration_controls[dimension][control]))
        return false;
    }
  }

  const std::vector<long double> vx_ay = multiplyBernsteinPolynomials(
      velocity_controls[0], acceleration_controls[1]);
  const std::vector<long double> vy_ax = multiplyBernsteinPolynomials(
      velocity_controls[1], acceleration_controls[0]);
  const std::vector<long double> vx_vx = multiplyBernsteinPolynomials(
      velocity_controls[0], velocity_controls[0]);
  const std::vector<long double> vy_vy = multiplyBernsteinPolynomials(
      velocity_controls[1], velocity_controls[1]);
  if (vx_ay.empty() || vx_ay.size() != vy_ax.size() || vx_vx.empty() ||
      vx_vx.size() != vy_vy.size())
    return false;

  std::vector<long double> numerator(vx_ay.size(), 0.0L);
  std::vector<long double> denominator(vx_vx.size(), 0.0L);
  for (std::size_t i = 0; i < numerator.size(); ++i) {
    numerator[i] = vx_ay[i] - vy_ax[i];
    if (!std::isfinite(numerator[i]))
      return false;
  }
  for (std::size_t i = 0; i < denominator.size(); ++i) {
    denominator[i] = vx_vx[i] + vy_vy[i] + static_cast<long double>(regularization);
    if (!std::isfinite(denominator[i]))
      return false;
  }

  struct BoundInterval {
    std::vector<long double> numerator;
    std::vector<long double> denominator;
    std::size_t depth;
  };
  std::vector<BoundInterval> pending;
  pending.push_back({numerator, denominator, 0});
  long double maximum_rate = 0.0L;
  constexpr std::size_t kMaxBoundIntervals = 131072;
  std::size_t processed_intervals = 0;
  while (!pending.empty()) {
    BoundInterval interval = std::move(pending.back());
    pending.pop_back();
    if (++processed_intervals > kMaxBoundIntervals)
      return false;
    if (interval.depth < subdivision_depth) {
      std::vector<long double> numerator_left, numerator_right;
      std::vector<long double> denominator_left, denominator_right;
      if (!subdivideBernsteinPolynomialHalf(
              interval.numerator, &numerator_left, &numerator_right) ||
          !subdivideBernsteinPolynomialHalf(
              interval.denominator, &denominator_left, &denominator_right))
        return false;
      pending.push_back(
          {std::move(numerator_right), std::move(denominator_right), interval.depth + 1});
      pending.push_back(
          {std::move(numerator_left), std::move(denominator_left), interval.depth + 1});
      continue;
    }

    long double numerator_upper = 0.0L;
    for (const long double coefficient : interval.numerator)
      numerator_upper = std::max(numerator_upper, std::abs(coefficient));
    long double denominator_lower = *std::min_element(
        interval.denominator.begin(), interval.denominator.end());
    const long double rounding_margin =
        64.0L * std::numeric_limits<long double>::epsilon();
    numerator_upper *= 1.0L + rounding_margin;
    denominator_lower -= std::abs(denominator_lower) * rounding_margin;
    denominator_lower = std::max(
        denominator_lower, static_cast<long double>(regularization));
    const long double interval_rate = numerator_upper / denominator_lower;
    if (!std::isfinite(interval_rate))
      return false;
    maximum_rate = std::max(maximum_rate, interval_rate);
  }

  maximum_rate *= 1.0L + 64.0L * std::numeric_limits<long double>::epsilon();
  if (!std::isfinite(maximum_rate) ||
      maximum_rate > static_cast<long double>(std::numeric_limits<double>::max()))
    return false;
  double rounded_bound = static_cast<double>(maximum_rate);
  if (rounded_bound > 0.0)
    rounded_bound = std::nextafter(rounded_bound, std::numeric_limits<double>::infinity());
  if (!std::isfinite(rounded_bound))
    return false;
  *yaw_rate_bound = rounded_bound;
  return true;
}

// Every accepted leaf emits its midpoint as well as both endpoints. This
// detects a curved segment whose endpoints happen to be close, while a depth,
// time, and total-sample bound makes pathological splines fail closed.
template <typename PositionEvaluator>
inline AdaptiveFinalTrajectorySampling makeAdaptiveFinalTrajectorySamples(
    const std::vector<double>& piece_durations,
    const std::vector<double>& piece_derivative_bounds,
    PositionEvaluator evaluate_position,
    double max_spatial_step, double max_time_step, double min_time_step,
    std::size_t max_depth, std::size_t max_samples)
{
  AdaptiveFinalTrajectorySampling sampling;
  if (piece_durations.empty() || piece_derivative_bounds.size() != piece_durations.size() ||
      !std::isfinite(max_spatial_step) ||
      !std::isfinite(max_time_step) || !std::isfinite(min_time_step) ||
      max_spatial_step <= 0.0 || max_time_step <= 0.0 || min_time_step <= 0.0 ||
      max_depth == 0 || max_samples == 0)
    return sampling;

  double total_duration = 0.0;
  for (std::size_t piece_index = 0; piece_index < piece_durations.size(); ++piece_index) {
    const double duration = piece_durations[piece_index];
    const double derivative_bound = piece_derivative_bounds[piece_index];
    if (!std::isfinite(duration) || duration <= 0.0 ||
        !std::isfinite(derivative_bound) || derivative_bound < 0.0)
      return sampling;
    total_duration += duration;
    if (!std::isfinite(total_duration))
      return sampling;
  }

  struct Interval {
    double start_time;
    double end_time;
    TrajectorySamplePoint start_position;
    TrajectorySamplePoint end_position;
    std::size_t depth;
  };

  const auto append_sample = [&sampling, max_samples](const AdaptiveTrajectorySample& sample) {
    if (!sampling.samples.empty()) {
      const AdaptiveTrajectorySample& previous = sampling.samples.back();
      if (previous.piece_index == sample.piece_index &&
          std::abs(previous.local_time - sample.local_time) <= 1e-12)
        return true;
    }
    if (sampling.samples.size() >= max_samples)
      return false;
    sampling.samples.push_back(sample);
    return true;
  };

  double piece_global_start = 0.0;
  for (std::size_t piece_index = 0; piece_index < piece_durations.size(); ++piece_index) {
    const double piece_duration = piece_durations[piece_index];
    const double piece_derivative_bound = piece_derivative_bounds[piece_index];
    const TrajectorySamplePoint start_position = evaluate_position(piece_index, 0.0);
    const TrajectorySamplePoint end_position = evaluate_position(piece_index, piece_duration);
    if (!trajectorySamplePointIsFinite(start_position) ||
        !trajectorySamplePointIsFinite(end_position)) {
      sampling.samples.clear();
      return sampling;
    }

    std::vector<Interval> pending;
    pending.push_back({0.0, piece_duration, start_position, end_position, 0});
    while (!pending.empty()) {
      const Interval interval = pending.back();
      pending.pop_back();
      const double interval_duration = interval.end_time - interval.start_time;
      const double midpoint_time = interval.start_time + 0.5 * interval_duration;
      const TrajectorySamplePoint midpoint_position =
          evaluate_position(piece_index, midpoint_time);
      if (!std::isfinite(interval_duration) || interval_duration <= 0.0 ||
          !trajectorySamplePointIsFinite(midpoint_position)) {
        sampling.samples.clear();
        return sampling;
      }

      const double bounded_arc_length = interval_duration * piece_derivative_bound;
      if (!std::isfinite(bounded_arc_length)) {
        sampling.samples.clear();
        return sampling;
      }
      const bool accepts_interval = interval_duration <= max_time_step &&
          bounded_arc_length <= max_spatial_step &&
          trajectorySampleDistance(interval.start_position, midpoint_position) <=
              max_spatial_step &&
          trajectorySampleDistance(midpoint_position, interval.end_position) <=
              max_spatial_step;
      if (accepts_interval) {
        if (!append_sample({piece_index, interval.start_time,
                piece_global_start + interval.start_time, interval.start_position}) ||
            !append_sample({piece_index, midpoint_time,
                piece_global_start + midpoint_time, midpoint_position}) ||
            !append_sample({piece_index, interval.end_time,
                piece_global_start + interval.end_time, interval.end_position})) {
          sampling.samples.clear();
          return sampling;
        }
        continue;
      }

      if (interval.depth >= max_depth || 0.5 * interval_duration < min_time_step ||
          pending.size() + 2 > max_samples) {
        sampling.samples.clear();
        return sampling;
      }
      // LIFO stack: right first so accepted leaves remain in chronological order.
      pending.push_back({midpoint_time, interval.end_time, midpoint_position,
          interval.end_position, interval.depth + 1});
      pending.push_back({interval.start_time, midpoint_time, interval.start_position,
          midpoint_position, interval.depth + 1});
    }
    piece_global_start += piece_duration;
  }

  if (sampling.samples.empty())
    return sampling;
  for (std::size_t i = 1; i < sampling.samples.size(); ++i) {
    const AdaptiveTrajectorySample& previous = sampling.samples[i - 1];
    const AdaptiveTrajectorySample& current = sampling.samples[i];
    const double dt = current.global_time - previous.global_time;
    if (!std::isfinite(dt) || dt < -1e-12 || dt > max_time_step + 1e-12 ||
        trajectorySampleDistance(previous.position, current.position) >
            max_spatial_step + 1e-12) {
      sampling.samples.clear();
      return sampling;
    }
  }
  sampling.valid = true;
  return sampling;
}

// The motion limit turns a time interval into a conservative arc-length
// interval.  Include both endpoints explicitly so a stopped final spline
// piece cannot avoid footprint validation.
inline FinalTrajectorySampling makeFinalTrajectorySampling(double duration, double speed_limit,
    double max_spatial_step, double max_time_step)
{
  FinalTrajectorySampling sampling;
  if (!std::isfinite(duration) || !std::isfinite(speed_limit) ||
      !std::isfinite(max_spatial_step) || !std::isfinite(max_time_step) || duration <= 0.0 ||
      speed_limit <= 0.0 || max_spatial_step <= 0.0 || max_time_step <= 0.0)
    return sampling;

  const double bounded_step = std::min(max_time_step, max_spatial_step / speed_limit);
  if (!std::isfinite(bounded_step) || bounded_step <= 0.0)
    return sampling;

  const double intervals_as_double = std::ceil(duration / bounded_step);
  constexpr std::size_t kMaxIntervals = 1000000;
  if (!std::isfinite(intervals_as_double) || intervals_as_double < 1.0 ||
      intervals_as_double > static_cast<double>(kMaxIntervals))
    return sampling;

  const std::size_t intervals = static_cast<std::size_t>(intervals_as_double);
  sampling.time_step = duration / static_cast<double>(intervals);
  sampling.times.reserve(intervals + 1);
  for (std::size_t i = 0; i < intervals; ++i)
    sampling.times.push_back(static_cast<double>(i) * sampling.time_step);
  sampling.times.push_back(duration);
  sampling.valid = true;
  return sampling;
}

inline double yawForTrajectorySample(double vx, double vy, double previous_yaw,
    double low_speed_threshold = 0.02)
{
  if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(previous_yaw) ||
      !std::isfinite(low_speed_threshold) || low_speed_threshold < 0.0)
    return previous_yaw;
  if (std::hypot(vx, vy) <= low_speed_threshold)
    return previous_yaw;
  return std::atan2(vy, vx);
}

enum class TerminalNavigationResult {
  NOT_TERMINAL,
  MISSION_FAILED,
  MISSION_SUCCEEDED,
};

inline TerminalNavigationResult terminalNavigationResult(int exploration_result)
{
  // FINAL_RESULT::NO_FRONTIER and FINAL_RESULT::REACH_OBJECT are intentionally
  // protocol values 3 and 4, respectively, shared with the supervisor.
  if (exploration_result == 3)
    return TerminalNavigationResult::MISSION_FAILED;
  if (exploration_result == 4)
    return TerminalNavigationResult::MISSION_SUCCEEDED;
  return TerminalNavigationResult::NOT_TERMINAL;
}

class ReplanSettleGate {
public:
  void reset() { stable_since_ = -1.0; }

  bool update(double now, double horizontal_speed, double speed_threshold, double stable_sec)
  {
    if (horizontal_speed > speed_threshold) {
      stable_since_ = -1.0;
      return false;
    }
    if (stable_since_ < 0.0) {
      stable_since_ = now;
      return stable_sec <= 0.0;
    }
    return now - stable_since_ >= stable_sec;
  }

private:
  double stable_since_ = -1.0;
};

}  // namespace apexnav_planner

#endif
