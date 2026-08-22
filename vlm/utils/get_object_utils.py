"""Object-map helpers backed exclusively by the Lite YOLOE segmentation service."""

import time

import cv2
import numpy as np

from vlm.detector.yoloe import YOLOEClient
from vlm.utils.get_itm_message import get_itm_message


yoloe_detector = YOLOEClient(port=12184)


def get_object_class_names(right_label, similar_answer):
    """Return target-first, de-duplicated names for ``MultipleMasksWithConfidence``."""
    targets, labels = _merge_labels(right_label, similar_answer)
    target_name = targets[0] if targets else "target"
    return [target_name] + [label for label in labels if label not in targets]


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    """Render one detection and return its model mask (or a box fallback)."""
    object_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bbox_denorm = detections.boxes[idx].detach().cpu().numpy() * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(value) for value in bbox_denorm]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1] - 1), min(y2, img.shape[0] - 1)
    if idx < len(detections.masks) and detections.masks[idx] is not None:
        object_mask = detections.masks[idx].astype(np.uint8)
    else:
        object_mask[y1 : y2 + 1, x1 : x2 + 1] = 1

    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(segmented_img, [contour], 0, color, 4)
    cv2.rectangle(segmented_img, (x1, y1), (x2, y2), color, 2)
    label_text = f"{label} ({score:.2f})"
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
    )
    label_y = y1 - text_height
    cv2.rectangle(
        segmented_img,
        (x1, label_y - 30),
        (x1 + text_width, label_y + text_height),
        color,
        2,
    )
    cv2.putText(
        segmented_img,
        label_text,
        (x1, label_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    return segmented_img, object_mask


def _get_yoloe_params(cfg):
    yoloe_cfg = getattr(cfg, "yoloe", None)
    if yoloe_cfg is None:
        raise ValueError("Lite VLM requires detector.yoloe configuration")
    return (
        getattr(yoloe_cfg, "confidence_threshold", 0.3),
        getattr(yoloe_cfg, "iou_threshold", 0.5),
        getattr(yoloe_cfg, "agnostic_nms", True),
    )


def _merge_labels(right_label, similar_answer):
    targets = [label.strip() for label in right_label.split("|") if label.strip()]
    all_labels = list(
        dict.fromkeys(
            targets + [str(label).strip() for label in similar_answer if str(label).strip()]
        )
    )
    return targets, all_labels


def get_object(right_label, img, cfg, similar_answer, return_stats=False):
    """Detect target/confusion objects while preserving the legacy four-value result."""
    score_list, object_masks_list, label_list = [], [], []
    segmented_img = img.copy()
    targets, all_labels = _merge_labels(right_label, similar_answer)
    conf_thres, iou_thres, agnostic_nms = _get_yoloe_params(cfg)
    request_start = time.perf_counter()
    detections = yoloe_detector.predict(
        img,
        classes=all_labels,
        agnostic_nms=agnostic_nms,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
    yoloe_latency_ms = (time.perf_counter() - request_start) * 1000.0

    for idx, label_detected in enumerate(detections.phrases):
        if label_detected not in all_labels:
            continue
        score = detections.logits[idx].item()
        color = (255, 0, 0) if label_detected in targets else (0, 255, 0)
        segmented_img, object_mask = get_segmentation(
            segmented_img, idx, detections, img, label_detected, score, color
        )
        score_list.append(float(score))
        object_masks_list.append(object_mask)
        label_list.append(
            0 if label_detected in targets else all_labels.index(label_detected) - len(targets) + 1
        )

    result = (segmented_img, score_list, object_masks_list, label_list)
    if return_stats:
        return result + ({"yoloe_latency_ms": yoloe_latency_ms},)
    return result


def get_object_with_itm(label, img, cfg):
    """Retain the combined helper using the same Lite YOLOE detector."""
    segmented_img, scores, masks, _ = get_object(label, img, cfg, [])
    cosine_list, itm_score_list = [], []
    for object_mask in masks:
        image_detected = crop_and_expand_box(img, object_mask)
        cosine, itm_score = get_itm_message(image_detected, label)
        cosine_list.append(cosine)
        itm_score_list.append(itm_score)
    return segmented_img, scores, masks, cosine_list, itm_score_list


def crop_and_expand_box(img, object_mask, expand_pixels=0.4):
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_min = max(int(x_min * (1 - expand_pixels)), 0)
    y_min = max(int(y_min * (1 - expand_pixels)), 0)
    x_max = min(int(x_max * (1 + expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max * (1 + expand_pixels)), img.shape[0] - 1)
    return img[y_min : y_max + 1, x_min : x_max + 1]
