#include <algorithm>
#include <cmath>
#include <plan_env/semantic_observability.h>

namespace apexnav_planner {
namespace semantic_observability {

double clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(max_value, value));
}

double heightAdaptation(double h, double mu_v, double sigma_v)
{
  const double sigma = std::max(sigma_v, 1e-3);
  const double diff = h - mu_v;
  return std::exp(-(diff * diff) / (2.0 * sigma * sigma));
}

double distanceQuality(double distance, double lambda_d)
{
  return std::exp(-std::max(0.0, lambda_d) * std::max(0.0, distance));
}

double angleQuality(double theta)
{
  const double c = std::cos(theta);
  return c * c;
}

double maskScaleQuality(double mask_scale, double r0)
{
  const double ref = std::max(r0, 1e-6);
  return clamp(mask_scale / ref, 0.0, 1.0);
}

double observability(double h, double mu_v, double sigma_v, double distance, double theta,
    double mask_scale, double lambda_d, double r0)
{
  return heightAdaptation(h, mu_v, sigma_v) * distanceQuality(distance, lambda_d) *
         angleQuality(theta) * maskScaleQuality(mask_scale, r0);
}

double saturatedEvidence(double observability_score, double confidence, int observation_num,
    double beta)
{
  const int n = std::max(0, observation_num);
  const double saturation = 1.0 - std::exp(-std::max(0.0, beta) * n);
  return observability_score * clamp(confidence, 0.0, 1.0) * saturation;
}

}  // namespace semantic_observability
}  // namespace apexnav_planner
