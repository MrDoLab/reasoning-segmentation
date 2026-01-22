"""
ReasonSeg Competition - Bbox Voting
MLLM 앙상블 bbox 투표 알고리즘
"""

import numpy as np
from typing import List, Dict

from src.config import VOTING_CONFIG
from src.utils import calculate_iou


def vote_bboxes(predictions: List[Dict]) -> Dict:
    """
    MLLM bbox 예측을 투표하여 최적 bbox 선정

    Args:
        predictions: [
            {'model': str, 'bbox': [x1,y1,x2,y2], 'confidence': float, 'reasoning': str},
            ...
        ]

    Returns:
        {
            'bbox': [x1, y1, x2, y2],
            'confidence': float,
            'strategy': str,
            'individual_preds': predictions
        }
    """
    if len(predictions) == 1:
        p = predictions[0]
        return {
            'bbox': p['bbox'],
            'confidence': p['confidence'],
            'strategy': 'single_model',
            'individual_preds': predictions,
        }

    strategy = VOTING_CONFIG['strategy']

    if strategy == 'weighted_average':
        return weighted_average_voting(predictions)
    if strategy == 'highest_conf':
        return highest_confidence_voting(predictions)
    if strategy == 'consensus':
        return consensus_voting(predictions)
    return weighted_average_voting(predictions)


def weighted_average_voting(predictions: List[Dict]) -> Dict:
    """Confidence 기반 가중 평균 투표"""
    confidences = np.array([p['confidence'] for p in predictions])
    min_conf = VOTING_CONFIG['min_confidence']
    valid_mask = confidences >= min_conf

    if not valid_mask.any():
        best_idx = confidences.argmax()
        return {
            'bbox': predictions[best_idx]['bbox'],
            'confidence': predictions[best_idx]['confidence'],
            'strategy': 'highest_conf (fallback)',
            'individual_preds': predictions,
        }

    weights = confidences[valid_mask]
    weights = weights / weights.sum()
    bboxes = np.array([p['bbox'] for p in predictions])[valid_mask]
    weighted_bbox = (bboxes * weights[:, None]).sum(axis=0)

    return {
        'bbox': weighted_bbox.tolist(),
        'confidence': float(confidences[valid_mask].mean()),
        'strategy': 'weighted_average',
        'individual_preds': predictions,
    }


def highest_confidence_voting(predictions: List[Dict]) -> Dict:
    """가장 높은 confidence bbox 선택"""
    confidences = [p['confidence'] for p in predictions]
    best_idx = int(np.argmax(confidences))

    return {
        'bbox': predictions[best_idx]['bbox'],
        'confidence': predictions[best_idx]['confidence'],
        'strategy': 'highest_confidence',
        'individual_preds': predictions,
    }


def consensus_voting(predictions: List[Dict]) -> Dict:
    """IoU 기반 합의 투표"""
    bboxes = [p['bbox'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]

    n = len(bboxes)
    iou_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(bboxes[i], bboxes[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    iou_threshold = VOTING_CONFIG['iou_threshold']
    avg_ious = iou_matrix.sum(axis=1) / (n - 1)
    valid_mask = avg_ious >= iou_threshold

    if not valid_mask.any():
        return weighted_average_voting(predictions)

    selected_bboxes = np.array(bboxes)[valid_mask]
    selected_confs = np.array(confidences)[valid_mask]
    weights = selected_confs / selected_confs.sum()
    consensus_bbox = (selected_bboxes * weights[:, None]).sum(axis=0)

    return {
        'bbox': consensus_bbox.tolist(),
        'confidence': float(selected_confs.mean()),
        'strategy': 'consensus',
        'individual_preds': predictions,
    }
