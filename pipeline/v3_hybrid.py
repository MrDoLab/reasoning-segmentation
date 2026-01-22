"""
v3 Hybrid Pipeline
query 키워드로 single/multi 자동 분기 → MLLM → SAM (multi는 mask merge)
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import IMAGE_SIZE, VERBOSE
from src.multi import combine_masks, is_multi_object_query
from src.prompts import fix_bbox_format


def initial_segmentation(
    sam_model,
    image: np.ndarray,
    bbox: List[float],
) -> Tuple[np.ndarray, float]:
    """SAM2.1 bbox 프롬프트로 마스크 생성"""
    sam_model.set_image(image)
    mask, score = sam_model.predict_with_bbox(bbox)
    return mask, score


def _process_single(
    mllm_model,
    sam_model,
    image: np.ndarray,
    query: str,
    verbose: bool,
) -> Dict:
    pred = mllm_model.predict(image, query, is_multi=False)

    if pred.get('bbox') is not None:
        bbox = pred['bbox']
    elif pred.get('bboxes'):
        bbox = pred['bboxes'][0]
    else:
        bbox = None

    bbox = fix_bbox_format(bbox, IMAGE_SIZE)

    if verbose:
        print(f"  [SINGLE] BBox: {bbox}")

    mask, sam_score = initial_segmentation(sam_model, image, bbox)

    if verbose:
        print(f"  SAM score: {sam_score:.3f}")

    return {
        'mllm_pred': {
            'bbox': bbox,
            'confidence': pred.get('confidence', 0.5),
            'raw_response': pred.get('raw_response', ''),
        },
        'sam_mask': mask,
        'sam_score': sam_score,
        'is_multi': False,
    }


def _process_multi(
    mllm_model,
    sam_model,
    image: np.ndarray,
    query: str,
    verbose: bool,
) -> Dict:
    pred = mllm_model.predict(image, query, is_multi=True)

    if verbose and pred.get('raw_response'):
        preview = pred['raw_response'][:150]
        print(f"  [MULTI] Raw response preview: {preview}...")

    if pred.get('bboxes'):
        bboxes = pred['bboxes']
    elif pred.get('bbox') is not None:
        bboxes = [pred['bbox']]
    else:
        bboxes = []

    fixed_bboxes = []
    for i, bbox in enumerate(bboxes):
        fixed = fix_bbox_format(bbox, IMAGE_SIZE)
        fixed_bboxes.append(fixed)
        if verbose:
            print(f"    Box {i + 1}: {fixed}")

    if not fixed_bboxes:
        if verbose:
            print("  No boxes detected, using center fallback")
        fixed_bboxes = [fix_bbox_format(None, IMAGE_SIZE)]

    if verbose:
        print(f"  Detected {len(fixed_bboxes)} objects")

    sam_model.set_image(image)
    individual_masks = []
    sam_scores = []

    for bbox in fixed_bboxes:
        mask, score = sam_model.predict_with_bbox(bbox)
        individual_masks.append(mask)
        sam_scores.append(score)

    if verbose:
        print(f"  SAM scores: {[f'{s:.3f}' for s in sam_scores]}")

    combined_mask = combine_masks(individual_masks)

    if verbose:
        print(f"  Combined mask coverage: {combined_mask.sum()} pixels")

    return {
        'mllm_pred': {
            'bboxes': fixed_bboxes,
            'confidence': pred.get('confidence', 0.6),
            'raw_response': pred.get('raw_response', ''),
        },
        'sam_mask': combined_mask,
        'sam_score': float(np.mean(sam_scores)) if sam_scores else 0.0,
        'sam_scores': sam_scores,
        'individual_masks': individual_masks,
        'is_multi': True,
    }


def run_hybrid(
    mllm_model: Any,
    sam_model,
    all_data: List[Dict],
    verbose: bool = None,
    unload_mllm: bool = True,
) -> List[Dict]:
    """
    Single/Multi hybrid 파이프라인

    Args:
        mllm_model: Qwen3Model instance
        sam_model: SAMModel instance
        all_data: [{'id', 'image', 'query', ...}, ...]
        verbose: 진행 출력 (None이면 src.config.VERBOSE)
        unload_mllm: MLLM VRAM 해제 여부

    Returns:
        all_data (in-place 업데이트 후 동일 참조)
    """
    if verbose is None:
        verbose = VERBOSE

    n = len(all_data)
    multi_count = sum(1 for item in all_data if is_multi_object_query(item['query']))

    if verbose:
        print(f"Hybrid pipeline: samples={n}, single={n - multi_count}, multi={multi_count}")

    stats = {'single': 0, 'multi': 0}

    for item in tqdm(all_data, desc="Hybrid pipeline", disable=not verbose):
        query = item['query']
        is_multi = is_multi_object_query(query)
        item['is_multi'] = is_multi

        mode_label = "MULTI" if is_multi else "SINGLE"
        if verbose:
            print(f"\n[{mode_label}] {item['id']}: {query[:60]}...")

        if is_multi:
            result = _process_multi(mllm_model, sam_model, item['image'], query, verbose)
            stats['multi'] += 1
        else:
            result = _process_single(mllm_model, sam_model, item['image'], query, verbose)
            stats['single'] += 1

        item['mllm_pred'] = result['mllm_pred']
        item['sam_mask'] = result['sam_mask']
        item['sam_score'] = result['sam_score']
        if result.get('individual_masks') is not None:
            item['individual_masks'] = result['individual_masks']

    if unload_mllm:
        if hasattr(mllm_model, 'model') and mllm_model.model is not None:
            del mllm_model.model
        if hasattr(mllm_model, 'processor') and mllm_model.processor is not None:
            del mllm_model.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print("MLLM unloaded, GPU memory freed")

    if verbose:
        print(f"\nHybrid pipeline completed: single={stats['single']}, multi={stats['multi']}")

    return all_data
