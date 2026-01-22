"""
Multi-object utilities for v3 hybrid pipeline.
"""

import re
from typing import Dict, List

import numpy as np

from src.config import HYBRID_CONFIG, IMAGE_SIZE


def is_multi_object_query(query: str) -> bool:
    """
    Query가 multiple objects를 요구하는지 확인.

    keyword_match (HYBRID_CONFIG):
      - 'word_boundary': unified pipeline (final Cell 2)
      - 'substring': Cell 0 prototype (any keyword in query)
    """
    query_lower = query.lower()
    match_mode = HYBRID_CONFIG.get('keyword_match', 'word_boundary')

    if match_mode == 'substring':
        return any(kw in query_lower for kw in HYBRID_CONFIG['multi_keywords'])

    padded = f" {query_lower} "
    for keyword in HYBRID_CONFIG['multi_keywords']:
        if ' ' in keyword:
            if keyword in query_lower:
                return True
        elif f" {keyword} " in padded:
            return True
        elif query_lower.startswith(keyword + " "):
            return True
        elif query_lower.endswith(" " + keyword):
            return True

    return False


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """여러 마스크를 하나로 합침 (Logical OR)."""
    if not masks:
        h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
        return np.zeros((h, w), dtype=bool)

    combined = masks[0].astype(bool)
    for mask in masks[1:]:
        combined = np.logical_or(combined, mask.astype(bool))

    return combined


def parse_multi_response(response: str) -> Dict:
    """Multiple bounding boxes 파싱."""
    result = {
        'bboxes': [],
        'confidence': 0.6,
        'reasoning': response,
    }

    box_pattern = r'Box\s+\d+:\s*\[([^\]]+)\]'
    matches = re.findall(box_pattern, response, re.IGNORECASE)

    if matches:
        for match in matches:
            try:
                coords = [float(x.strip()) for x in match.split(',')]
                if len(coords) == 4:
                    result['bboxes'].append(coords)
            except ValueError:
                continue

    if not result['bboxes']:
        numbered_pattern = r'\d+[\.\)]\s*\[([^\]]+)\]'
        matches = re.findall(numbered_pattern, response)
        for match in matches:
            try:
                coords = [float(x.strip()) for x in match.split(',')]
                if len(coords) == 4:
                    result['bboxes'].append(coords)
            except ValueError:
                continue

    if not result['bboxes']:
        coord_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
        matches = re.findall(coord_pattern, response)
        for match in matches:
            try:
                result['bboxes'].append([float(x) for x in match])
            except ValueError:
                continue

    return result
