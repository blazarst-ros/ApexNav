from typing import Iterable, List

import rospy

from llm.answer_reader.structured_feedback import read_latest_feedback_record
from llm.utils.semantic_prior_defaults import ensure_semantic_priors


def _category_list(target_label: str, confusion_labels: Iterable[str]) -> List[str]:
    categories = [str(target_label)]
    categories.extend(str(label) for label in confusion_labels if isinstance(label, str))
    return categories


def publish_semantic_priors_to_ros(llm_answer_path: str, target_label: str, confusion_labels):
    """Publish category priors as index-addressable ROS params for C++ map code."""
    categories = _category_list(target_label, confusion_labels)
    record = read_latest_feedback_record(llm_answer_path, target_label)
    parsed_priors = {}
    if record:
        parsed_priors = record.get("semantic_verification_prior", {})

    priors = ensure_semantic_priors(target_label, categories[1:], parsed_priors)
    rospy.set_param("/semantic_prior/categories", categories)

    for idx, category in enumerate(categories):
        prior = priors[category]
        base = f"/semantic_prior/label_{idx}"
        rospy.set_param(f"{base}/category", category)
        rospy.set_param(f"{base}/mu_v", float(prior["mu_v"]))
        rospy.set_param(f"{base}/sigma_v", float(prior["sigma_v"]))


def compute_mask_scales(object_masks_list):
    scales = []
    for mask in object_masks_list:
        if mask is None or not hasattr(mask, "size") or mask.size == 0:
            scales.append(0.0)
        else:
            scales.append(float((mask > 0).sum()) / float(mask.size))
    return scales
