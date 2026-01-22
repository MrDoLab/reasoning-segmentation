"""
v1 Refinement Pipeline
MLLM → SAM → verify → feedback → 재시도 (optional)
"""

import numpy as np
from typing import List, Dict, Tuple

from src.config import REFINEMENT_CONFIG
from src.utils import set_seed
from src.verify import batch_verify_segmentation


def initial_segmentation(
    sam_model,
    image: np.ndarray,
    bbox: List[float],
) -> Tuple[np.ndarray, float]:
    """SAM2.1 bbox 프롬프트로 마스크 생성"""
    sam_model.set_image(image)
    mask, score = sam_model.predict_with_bbox(bbox)
    return mask, score


def run_refinement(
    mllm_model,
    sam_model,
    images: List[np.ndarray],
    queries: List[str],
    image_ids: List[str] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Batch Iterative Refinement Pipeline with Verification

    Returns:
        [{
            'final_bbox': [x1, y1, x2, y2],
            'final_mask': (H, W) bool array,
            'final_score': float,
            'rounds': int,
            'verification_history': [...]
        }, ...]
    """
    n_images = len(images)

    if image_ids is None:
        image_ids = [f"image_{i}" for i in range(n_images)]

    if not REFINEMENT_CONFIG['enabled']:
        if verbose:
            print("Refinement disabled. Running basic pipeline...")

        results = []
        preds = mllm_model.predict_batch(images, queries)

        for image, pred in zip(images, preds):
            mask, score = initial_segmentation(sam_model, image, pred['bbox'])
            results.append({
                'final_bbox': pred['bbox'],
                'final_mask': mask,
                'final_score': score,
                'rounds': 1,
                'verification_history': [],
            })

        return results

    max_rounds = REFINEMENT_CONFIG['max_rounds']
    base_seed = REFINEMENT_CONFIG['base_seed']
    comparison_type = REFINEMENT_CONFIG['comparison_type']
    early_stop = REFINEMENT_CONFIG['early_stop']

    final_results = [None] * n_images
    active_indices = list(range(n_images))
    current_bboxes = [None] * n_images
    current_masks = [None] * n_images
    current_scores = [0.0] * n_images
    verification_histories = [[] for _ in range(n_images)]
    feedbacks = [None] * n_images

    for round_num in range(1, max_rounds + 1):
        if not active_indices:
            if verbose:
                print("All images verified successfully!")
            break

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Refinement Round {round_num}/{max_rounds}")
            print(f"   Processing {len(active_indices)}/{n_images} images")
            print(f"{'=' * 80}")

        round_seed = base_seed + round_num - 1
        set_seed(round_seed)

        if verbose:
            print(f"Seed set to {round_seed}")
            if round_num == 1:
                print(f"MLLM batch prediction ({len(active_indices)} images)...")
            else:
                print(f"MLLM re-prediction with feedback ({len(active_indices)} images)...")
                if active_indices and feedbacks[active_indices[0]]:
                    print(f"\nExample feedback for [{image_ids[active_indices[0]]}]:")
                    print(feedbacks[active_indices[0]][:200] + "...\n")

        batch_images = [images[i] for i in active_indices]
        batch_queries = [queries[i] for i in active_indices]
        batch_feedbacks = [feedbacks[i] for i in active_indices]

        preds = mllm_model.predict_batch(batch_images, batch_queries, batch_feedbacks)

        for idx, pred in zip(active_indices, preds):
            current_bboxes[idx] = pred['bbox']
            if verbose:
                print(
                    f"  [{image_ids[idx]}] BBox: {pred['bbox'][:2]}...{pred['bbox'][2:]}, "
                    f"Conf: {pred['confidence']:.2f}"
                )

        if verbose:
            print(f"SAM segmentation ({len(active_indices)} images)...")

        for idx in active_indices:
            current_masks[idx], current_scores[idx] = initial_segmentation(
                sam_model, images[idx], current_bboxes[idx]
            )

        if verbose:
            for idx in active_indices:
                mask_coverage = (current_masks[idx].sum() / current_masks[idx].size) * 100
                print(
                    f"  [{image_ids[idx]}] Coverage: {mask_coverage:.2f}%, "
                    f"Score: {current_scores[idx]:.2f}"
                )

        if verbose:
            print(f"Verifying segmentation ({len(active_indices)} images)...")

        verifications = batch_verify_segmentation(
            mllm_model,
            [images[i] for i in active_indices],
            [current_masks[i] for i in active_indices],
            [current_bboxes[i] for i in active_indices],
            [queries[i] for i in active_indices],
            comparison_type=comparison_type,
        )

        next_round_indices = []

        for i, idx in enumerate(active_indices):
            verification = verifications[i]

            verification_histories[idx].append({
                'round': round_num,
                'bbox': current_bboxes[idx],
                'mask': current_masks[idx].copy(),
                'mask_score': current_scores[idx],
                'verification': verification,
            })

            if verbose:
                status = "PASS" if verification['matches'] else "FAIL"
                print(f"  [{image_ids[idx]}] {status} - {verification['identified_object'][:30]}...")

            if verification['matches'] and early_stop:
                final_results[idx] = {
                    'final_bbox': current_bboxes[idx],
                    'final_mask': current_masks[idx],
                    'final_score': current_scores[idx],
                    'rounds': round_num,
                    'verification_history': verification_histories[idx],
                }
            else:
                reasoning = verification.get('reasoning', 'The segmentation was incorrect.')
                identified = verification.get('identified_object', 'Unknown')
                bbox = current_bboxes[idx]

                feedback = (
                    "ATTENTION: Your previous prediction was INCORRECT!\n\n"
                    f"Round {round_num} Attempt:\n"
                    f"  - Your BBox: {bbox}\n"
                    f"  - What you found: {identified}\n"
                    f"  - Why it failed: {reasoning[:250]}\n\n"
                    "CRITICAL: You MUST predict a DIFFERENT location.\n"
                    f"Do NOT repeat the same bounding box "
                    f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}].\n"
                    "Try searching in a completely different area of the image."
                )
                feedbacks[idx] = feedback
                next_round_indices.append(idx)

                if verbose:
                    print("      -> Will re-predict with feedback in next round")

        active_indices = next_round_indices

        if round_num == max_rounds and active_indices:
            if verbose:
                print(f"Max rounds reached. Finalizing {len(active_indices)} remaining images...")

            for idx in active_indices:
                final_results[idx] = {
                    'final_bbox': current_bboxes[idx],
                    'final_mask': current_masks[idx],
                    'final_score': current_scores[idx],
                    'rounds': max_rounds,
                    'verification_history': verification_histories[idx],
                }

    if verbose:
        completed_early = sum(1 for r in final_results if r['rounds'] < max_rounds)
        print(f"\n{'=' * 80}")
        print("Batch refinement complete!")
        print(f"   Early stop: {completed_early}/{n_images}")
        print(f"   Max rounds: {n_images - completed_early}/{n_images}")
        print(f"{'=' * 80}\n")

    return final_results


def apply_refinement_results(all_data: List[Dict], results: List[Dict]) -> None:
    """run_refinement 결과를 all_data 항목에 반영"""
    for i, item in enumerate(all_data):
        item['mllm_pred'] = {
            'bbox': results[i]['final_bbox'],
            'confidence': 0.5,
        }
        item['sam_mask'] = results[i]['final_mask']
        item['sam_score'] = results[i]['final_score']
        item['refinement_rounds'] = results[i]['rounds']
        item['refinement_result'] = results[i]
