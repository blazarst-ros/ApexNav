#ifndef _SEMANTIC_OBSERVABILITY_H_
#define _SEMANTIC_OBSERVABILITY_H_

namespace apexnav_planner {
namespace semantic_observability {

double clamp(double value, double min_value, double max_value);
double heightAdaptation(double h, double mu_v, double sigma_v);
double distanceQuality(double distance, double lambda_d);
double angleQuality(double theta);
double maskScaleQuality(double mask_scale, double r0, double mask_sigmoid_k);
double observability(double h, double mu_v, double sigma_v, double distance, double theta,
    double mask_scale, double lambda_d, double r0, double mask_sigmoid_k);
double saturatedEvidence(double observability_score, double confidence, int observation_num,
    double beta);

}  // namespace semantic_observability
}  // namespace apexnav_planner

#endif
