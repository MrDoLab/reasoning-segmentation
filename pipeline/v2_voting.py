"""
v2 Voting Pipeline
MLLM (1~N) → vote → SAM
단일 모델이면 vote 스킵
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import VERBOSE
from src.voting import vote_bboxes


def initial_segmentation(
    sam_model,
    image: np.ndarray,
    bbox: List[float],
) -> Tuple[np.ndarray, float]:
    """SAM2.1 bbox 프롬프트로 마스크 생성"""
    sam_model.set_image(image)
    mask, score = sam_model.predict_with_bbox(bbox)
    return mask, score


def _predict_batch(model, images, queries):
    """배치 predict (Qwen3Model 등 predict_batch 지원 모델)"""
    return model.predict_batch(images, queries)


def run_voting(
    mllm_models: List[Tuple[str, Any]],
    sam_model,
    all_data: List[Dict],
    batch_size: int = 8,
    verbose: bool = None,
    unload_mllm: bool = True,
) -> List[Dict]:
    """
    MLLM voting → SAM 파이프라인

    Args:
        mllm_models: [('qwen3', model), ('llava', model), ...]
        sam_model: SAMModel instance
        all_data: [{'id', 'image', 'query', ...}, ...]
        batch_size: MLLM 배치 크기
        verbose: 진행 출력 (None이면 src.config.VERBOSE)
        unload_mllm: MLLM VRAM 해제 여부

    Returns:
        all_data (in-place 업데이트 후 동일 참조)
    """
    if verbose is None:
        verbose = VERBOSE

    n = len(all_data)
    num_batches = (n + batch_size - 1) // batch_size

    if verbose:
        names = [name for name, _ in mllm_models]
        print(f"Voting pipeline: models={names}, samples={n}, batch_size={batch_size}")

    # model_name -> list of per-sample predictions (index aligned with all_data)
    all_model_preds: Dict[str, List[Dict]] = {
        name: [None] * n for name, _ in mllm_models
    }

    for batch_start in tqdm(range(0, n, batch_size), desc="MLLM batches", disable=not verbose):
        batch_end = min(batch_start + batch_size, n)
        batch_items = all_data[batch_start:batch_end]
        batch_images = [item['image'] for item in batch_items]
        batch_queries = [item['query'] for item in batch_items]

        for model_name, model in mllm_models:
            batch_results = _predict_batch(model, batch_images, batch_queries)
            for local_i, pred in enumerate(batch_results):
                global_i = batch_start + local_i
                all_model_preds[model_name][global_i] = {
                    'model': model_name,
                    'bbox': pred['bbox'],
                    'confidence': pred['confidence'],
                    'reasoning': pred.get('reasoning', ''),
                }

    if unload_mllm:
        for _, model in mllm_models:
            if hasattr(model, 'model') and model.model is not None:
                del model.model
            if hasattr(model, 'processor') and model.processor is not None:
                del model.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print("MLLM unloaded, GPU memory freed")

    if verbose:
        print(f"SAM segmentation ({n} images)...")

    for i, item in enumerate(tqdm(all_data, desc="SAM segmentation", disable=not verbose)):
        predictions = [all_model_preds[name][i] for name, _ in mllm_models]
        vote_result = vote_bboxes(predictions)

        mask, score = initial_segmentation(sam_model, item['image'], vote_result['bbox'])

        item['mllm_pred'] = {
            'bbox': vote_result['bbox'],
            'confidence': vote_result['confidence'],
            'strategy': vote_result['strategy'],
            'individual_preds': vote_result['individual_preds'],
        }
        item['sam_mask'] = mask
        item['sam_score'] = score

    if verbose:
        print(f"Voting pipeline completed for {n} images")

    return all_data
