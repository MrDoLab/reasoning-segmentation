"""
ReasonSeg Competition - Utility Functions
유틸리티 함수 모음: Seed 고정, RLE 인코딩, BBox 처리, 시각화
"""

import random
import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Union
from pathlib import Path


# Seed 고정

def set_seed(seed: int = 42):
    """
    재현성을 위한 Seed 고정
    경진대회 필수 사항
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed fixed to {seed}")


# RLE 인코딩

def rle_encode(mask_tensor: Union[np.ndarray, torch.Tensor]) -> str:
    """
    마스크를 RLE(Run Length Encoding) 문자열로 변환
    
    Args:
        mask_tensor: [H, W] 또는 [B, H, W] 형태의 bool 타입 텐서
    
    Returns:
        단일 입력: RLE 문자열
        배치 입력: RLE 문자열 리스트
    """
    # 텐서를 CPU의 numpy 배열로 변환
    if torch.is_tensor(mask_tensor):
        mask = mask_tensor.detach().cpu().numpy()
    else:
        mask = mask_tensor

    # 입력 차원 확인
    if mask.ndim == 3:
        return [encode_single_mask(m) for m in mask]
    else:
        return encode_single_mask(mask)


def encode_single_mask(mask: np.ndarray) -> str:
    """단일 마스크(H, W)를 RLE 문자열로 변환"""
    # 픽셀을 일렬로 세움
    pixels = mask.flatten()

    # 변화 지점을 찾기 위해 앞뒤에 0을 붙임
    pixels = np.concatenate([[False], pixels, [False]])

    # 0 -> 1로 변하는 지점(시작)과 1 -> 0으로 변하는 지점(끝) 찾기
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # '끝 지점'에서 '시작 지점'을 빼서 '길이' 계산
    runs[1::2] -= runs[::2]

    # 결과 문자열 생성
    return ' '.join(str(x) for x in runs)


# Bounding Box 유틸리티

def normalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    Bbox 좌표를 정규화 [0, 1]
    
    Args:
        bbox: [x1, y1, x2, y2] (pixel coordinates)
        image_size: (width, height)
    
    Returns:
        normalized_bbox: [x1, y1, x2, y2] (0~1 range)
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox
    return [x1/w, y1/h, x2/w, y2/h]


def denormalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[int]:
    """
    정규화된 Bbox를 픽셀 좌표로 변환
    
    Args:
        bbox: [x1, y1, x2, y2] (normalized 0~1)
        image_size: (width, height)
    
    Returns:
        pixel_bbox: [x1, y1, x2, y2] (pixel coordinates)
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox
    return [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    두 Bbox 간 IoU (Intersection over Union) 계산
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2]
    
    Returns:
        iou: 0~1 사이 값
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection 계산
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union 계산
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def bbox_to_point(bbox: List[float]) -> Tuple[float, float]:
    """
    Bbox 중심점 계산
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# 시각화 유틸리티

def visualize_prediction(
    image: np.ndarray,
    bbox: List[int] = None,
    mask: np.ndarray = None,
    points: np.ndarray = None,
    save_path: str = None
):
    """
    예측 결과 시각화
    
    Args:
        image: (H, W, 3) RGB image
        bbox: [x1, y1, x2, y2] or None
        mask: (H, W) bool mask or None
        points: (N, 2) [[x, y], ...] or None
        save_path: 저장 경로
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    # Bbox 그리기
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    # Mask 오버레이
    if mask is not None:
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [0, 255, 0]
        ax.imshow(colored_mask, alpha=0.4)
    
    # Points 그리기
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=50, marker='o')
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 파일 I/O

def create_directories(dirs: Dict[str, str]):
    """필요한 디렉토리 생성"""
    for name, path in dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created {len(dirs)} directories")


def load_image(image_path: str) -> np.ndarray:
    """
    이미지 로드
    
    Args:
        image_path: 이미지 경로
    
    Returns:
        image: (H, W, 3) RGB numpy array
    """
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# 시각화

def visualize_result(item: Dict, figsize: Tuple[int, int] = (12, 4)):
    """
    결과 시각화: 원본, BBox, SAM만
    
    Args:
        item: {
            'id': str,
            'image': np.ndarray (H, W, 3),
            'query': str,
            'mllm_pred': {'bbox': [x1, y1, x2, y2], 'confidence': float} or None,
            'sam_mask': np.ndarray (H, W) bool or None,
            'sam_score': float (optional)
        }
        figsize: (width, height) figure size
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 원본 이미지
    axes[0].imshow(item['image'])
    axes[0].set_title(f"Original\n{item['id']}")
    axes[0].axis('off')
    
    # 2. MLLM BBox
    axes[1].imshow(item['image'])
    if item['mllm_pred'] and item['mllm_pred']['bbox']:
        bbox = item['mllm_pred']['bbox']
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].set_title(f"MLLM BBox\nConf: {item['mllm_pred']['confidence']:.2f}")
    axes[1].axis('off')
    
    # 3. SAM Mask
    axes[2].imshow(item['image'])
    if item['sam_mask'] is not None:
        axes[2].imshow(item['sam_mask'], alpha=0.5, cmap='jet')
        mask_pixels = item['sam_mask'].sum()
        total_pixels = item['sam_mask'].size
        coverage = mask_pixels / total_pixels * 100
        axes[2].set_title(f"SAM Mask\nScore: {item.get('sam_score', 0):.2f}\nCoverage: {coverage:.2f}%")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 추가 정보 출력
    print(f"\n{'='*60}")
    print(f"ID: {item['id']}")
    print(f"Query: {item['query']}")
    if item['mllm_pred']:
        print(f"BBox: {item['mllm_pred']['bbox']}")
        print(f"Confidence: {item['mllm_pred']['confidence']:.2f}")
    if item['sam_mask'] is not None:
        mask_pixels = item['sam_mask'].sum()
        total_pixels = item['sam_mask'].size
        coverage = mask_pixels / total_pixels * 100
        print(f"SAM mask shape: {item['sam_mask'].shape}")
        print(f"SAM mask pixels: {mask_pixels} / {total_pixels} ({coverage:.2f}%)")
        print(f"SAM score: {item.get('sam_score', 'N/A')}")
    print(f"{'='*60}\n")


def visualize_refinement_progress(item: Dict, result: Dict):
    """
    Refinement 진행 과정 시각화 (Round별 비교)
    
    Args:
        item: {
            'id': str,
            'image': np.ndarray (H, W, 3),
            'query': str
        }
        result: batch_iterative_refinement_pipeline의 결과
            {
                'final_bbox': [...],
                'final_mask': ...,
                'rounds': int,
                'verification_history': [
                    {
                        'round': int,
                        'bbox': [...],
                        'mask_score': float,
                        'verification': {...}
                    }, ...
                ]
            }
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    history = result['verification_history']
    n_rounds = len(history)
    
    if n_rounds == 0:
        print(f"⚠️ No refinement history for {item['id']}")
        return
    
    # 각 라운드마다 3개 subplot (Original | BBox | Mask)
    fig, axes = plt.subplots(n_rounds, 3, figsize=(12, 4 * n_rounds))
    
    # 1 round만 있을 경우 2D array로 변환
    if n_rounds == 1:
        axes = axes.reshape(1, -1)
    
    for i, round_info in enumerate(history):
        round_num = round_info['round']
        bbox = round_info['bbox']
        verification = round_info['verification']
        
        # 저장된 마스크 사용
        mask = round_info.get('mask', None)
        
        # 마스크가 없으면 최종 마스크 사용 (마지막 라운드)
        if mask is None and i == n_rounds - 1:
            mask = result['final_mask']
        
        # 상태 표시
        status = "✅ PASS" if verification.get('matches', False) else "❌ FAIL"
        
        # 1. 원본
        axes[i, 0].imshow(item['image'])
        axes[i, 0].set_title(f"Round {round_num}\n{item['id']}")
        axes[i, 0].axis('off')
        
        # 2. BBox
        axes[i, 1].imshow(item['image'])
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[i, 1].add_patch(rect)
        axes[i, 1].set_title(f"BBox\n{status}")
        axes[i, 1].axis('off')
        
        # 3. Mask (마지막 라운드만)
        axes[i, 2].imshow(item['image'])
        if mask is not None:
            axes[i, 2].imshow(mask, alpha=0.5, cmap='jet')
            coverage = (mask.sum() / mask.size) * 100
            axes[i, 2].set_title(f"SAM Result\nCoverage: {coverage:.2f}%")
        else:
            axes[i, 2].set_title("SAM Result\n(intermediate)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 상세 정보 출력
    print(f"\n{'='*80}")
    print(f"🔄 Refinement Progress: {item['id']}")
    print(f"Query: {item['query']}")
    print(f"Total Rounds: {n_rounds}")
    print(f"{'='*80}")
    
    for i, round_info in enumerate(history):
        round_num = round_info['round']
        bbox = round_info['bbox']
        verification = round_info['verification']
        
        status = "✅ PASS" if verification.get('matches', False) else "❌ FAIL"
        
        print(f"\nRound {round_num}: {status}")
        print(f"  BBox: {bbox}")
        print(f"  Identified: {verification.get('identified_object', 'N/A')}")
        print(f"  Matches: {verification.get('matches', False)}")
        if verification.get('corrected_bbox'):
            print(f"  Corrected BBox: {verification['corrected_bbox']}")
        print(f"  Reasoning: {verification.get('reasoning', 'N/A')[:100]}...")
    
    print(f"\n{'='*80}\n")
