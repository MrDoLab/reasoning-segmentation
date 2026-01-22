"""
ReasonSeg Competition - Verification Module
마스크 검증 및 반복적 개선을 위한 시각적 피드백 생성
"""

import numpy as np
import cv2
import re
from typing import Dict, Tuple, List
from PIL import Image


# 시각적 피드백 이미지 생성

def create_mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    마스크를 이미지에 반투명 오버레이
    
    Args:
        image: (H, W, 3) RGB numpy array
        mask: (H, W) bool array
        color: RGB 색상, 기본값 녹색
    
    Returns:
        overlay_image: (H, W, 3) RGB numpy array
    """
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask] = color
    overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
    return overlay


def create_comparison_image(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    원본 | 마스크 오버레이를 나란히 배치 (Side-by-Side)
    
    Args:
        original: (H, W, 3) RGB array
        mask: (H, W) bool array
    
    Returns:
        comparison: (H, W*2, 3) RGB array
    """
    # 마스크 오버레이 생성
    overlay = create_mask_overlay(original, mask)
    
    # 나란히 배치
    h, w = original.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = original      # 왼쪽: 원본
    comparison[:, w:] = overlay       # 오른쪽: 마스크
    
    # 구분선 추가
    comparison[:, w-2:w+2] = [255, 255, 255]
    
    # 텍스트 레이블 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comparison, "Segmentation", (w+10, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    return comparison


def create_triple_view(original: np.ndarray, mask: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    원본 | BBox 표시 | 마스크 오버레이 (3-way 비교)
    
    Args:
        original: (H, W, 3) RGB array
        mask: (H, W) bool array
        bbox: [x1, y1, x2, y2]
    
    Returns:
        triple: (H, W*3, 3) RGB array
    """
    # 1. 원본
    view1 = original.copy()
    
    # 2. BBox 표시
    view2 = original.copy()
    x1, y1, x2, y2 = [int(x) for x in bbox]
    cv2.rectangle(view2, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    # 3. 마스크 오버레이
    view3 = create_mask_overlay(original, mask)
    
    # 나란히 배치
    h, w = original.shape[:2]
    triple = np.zeros((h, w*3, 3), dtype=np.uint8)
    triple[:, :w] = view1
    triple[:, w:2*w] = view2
    triple[:, 2*w:] = view3
    
    # 구분선
    triple[:, w-2:w+2] = [255, 255, 255]
    triple[:, 2*w-2:2*w+2] = [255, 255, 255]
    
    # 레이블
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(triple, "Original", (10, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(triple, "Predicted BBox", (w+10, 40), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(triple, "SAM Result", (2*w+10, 40), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    return triple


# 검증 프롬프트 생성

def create_verification_prompt(query: str, comparison_type: str = "side_by_side") -> str:
    """
    검증 프롬프트 생성
    
    Args:
        query: 원본 쿼리
        comparison_type: "side_by_side" or "triple"
    
    Returns:
        prompt: 검증용 프롬프트
    """
    if comparison_type == "side_by_side":
        prompt = f"""
The image shows two views side-by-side:
- LEFT: Original image
- RIGHT: Segmentation result (colored mask highlights the segmented object)

Original task: {query}

Please analyze:
Step 1: What object is highlighted by the colored mask on the RIGHT side?
Object: <describe what you see in the highlighted area>

Step 2: Does this correctly answer the task "{query}"?
Answer: Yes or No

Step 3: Provide your reasoning
Reasoning: <explain why it's correct or what's wrong>
"""
    else:  # triple
        prompt = f"""
Three views are shown from left to right:
1. LEFT: Original image
2. MIDDLE: Predicted bounding box (red rectangle)
3. RIGHT: SAM segmentation result (colored mask)

Original task: {query}

Please analyze:
Step 1: What object is inside the red box in the MIDDLE image?
Object in BBox: <describe>

Step 2: What object is highlighted by the colored mask in the RIGHT image?
Object in Mask: <describe>

Step 3: Do they match each other and correctly answer the task "{query}"?
Answer: Yes or No

Step 4: Provide your reasoning
Reasoning: <explain your assessment>
"""
    
    return prompt


def parse_verification_response(response: str) -> Dict:
    """
    검증 응답 파싱
    
    Args:
        response: 모델 응답 텍스트
    
    Returns:
        parsed: {
            'identified_object': str,
            'matches': bool,
            'reasoning': str,
            'corrected_bbox': [x1, y1, x2, y2] or None,
            'confidence': float
        }
    """
    result = {
        'identified_object': 'Unknown',
        'matches': False,  # 기본값을 False로 (보수적)
        'reasoning': response,
        'corrected_bbox': None,
        'confidence': 0.5
    }
    
    # Object 추출 - 엄격한 패턴 사용 (질문 완전 배제)
    # 우선순위: "Object in Mask:" > "Object in BBox:" > 줄 시작 "Object:"
    object_patterns = [
        r'Object in Mask:\s*([^\n]+)',            # 최우선: 정확히 "Object in Mask:"
        r'Object in BBox:\s*([^\n]+)',            # "Object in BBox:"
        r'\n\s*Object:\s*([^\n]+)',               # 줄바꿈 후 "Object:" (답변)
        r'^Object:\s*([^\n]+)',                   # 줄 시작 "Object:" (답변)
    ]
    for pattern in object_patterns:
        object_match = re.search(pattern, response, re.MULTILINE)
        if object_match:
            obj_text = object_match.group(1).strip()
            # 간단한 필터링: 너무 길거나 질문 형태 배제
            if len(obj_text) < 200 and '?' not in obj_text:
                result['identified_object'] = obj_text
                break
    
    # Answer (Yes/No) 추출 - 더 robust하게
    # 패턴 1: "Answer: Yes" or "Answer: No"
    answer_patterns = [
        r'Answer[:\s]+(Yes|No)',
        r'Step 3[:\s]+.*?(Yes|No)',
        r'Do they match.*?\n\s*(Yes|No)'
    ]
    
    for pattern in answer_patterns:
        answer_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).lower()
            result['matches'] = (answer == 'yes')
            break
    
    # Reasoning 추출
    reasoning_match = re.search(r'Reasoning[:\s]+([^\n]+(?:\n(?!Step|Corrected|Confidence)[^\n]+)*)', response, re.IGNORECASE)
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()
    
    # Corrected BBox 추출
    bbox_patterns = [
        r'Corrected BBox[:\s]*\[([^\]]+)\]',
        r'Corrected[:\s]*\[([^\]]+)\]',
        r'\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]'
    ]
    
    for pattern in bbox_patterns:
        bbox_match = re.search(pattern, response, re.IGNORECASE)
        if bbox_match:
            try:
                if ',' in bbox_match.group(1):
                    coords = [float(x.strip()) for x in bbox_match.group(1).split(',')]
                else:
                    coords = [float(bbox_match.group(i)) for i in range(1, 5)]
                
                if len(coords) == 4:
                    result['corrected_bbox'] = coords
                    break
            except:
                continue
    
    # Confidence 추출
    conf_match = re.search(r'Confidence[:\s]*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE)
    if conf_match:
        try:
            result['confidence'] = float(conf_match.group(1))
        except:
            pass
    
    return result


# 검증 실행

def verify_segmentation(
    model,
    image: np.ndarray,
    mask: np.ndarray,
    bbox: List[float],
    query: str,
    comparison_type: str = "triple"
) -> Dict:
    """
    세그멘테이션 결과 검증
    
    Args:
        model: MLLM 모델 instance
        image: (H, W, 3) RGB array
        mask: (H, W) bool array
        bbox: [x1, y1, x2, y2]
        query: 원본 쿼리
        comparison_type: "side_by_side" or "triple"
    
    Returns:
        verification_result: {
            'identified_object': str,
            'matches': bool,
            'reasoning': str,
            'corrected_bbox': [...] or None,
            'confidence': float
        }
    """
    # 비교 이미지 생성
    if comparison_type == "triple":
        comparison_image = create_triple_view(image, mask, bbox)
    else:
        comparison_image = create_comparison_image(image, mask)
    
    # 검증 프롬프트
    prompt = create_verification_prompt(query, comparison_type)
    
    # PIL Image로 변환
    pil_image = Image.fromarray(comparison_image)
    
    # MLLM 예측
    response = model.predict(comparison_image, prompt)
    
    # 응답이 Dict가 아니라 str인 경우 처리
    if isinstance(response, dict):
        generated_text = response.get('reasoning', str(response))
    else:
        generated_text = str(response)
    
    # 파싱
    result = parse_verification_response(generated_text)
    
    return result


def batch_verify_segmentation(
    model,
    images: List[np.ndarray],
    masks: List[np.ndarray],
    bboxes: List[List[float]],
    queries: List[str],
    comparison_type: str = "triple"
) -> List[Dict]:
    """
    배치로 여러 세그멘테이션 검증 (진짜 배치 처리!)
    
    Args:
        model: MLLM 모델 instance
        images: [(H, W, 3), ...] RGB arrays
        masks: [(H, W), ...] bool arrays
        bboxes: [[x1, y1, x2, y2], ...]
        queries: [str, ...]
        comparison_type: "side_by_side" or "triple"
    
    Returns:
        results: [{...}, ...]
    """
    # Step 1: 모든 비교 이미지 생성
    comparison_images = []
    for i, (image, mask, bbox) in enumerate(zip(images, masks, bboxes)):
        if comparison_type == "triple":
            comp_img = create_triple_view(image, mask, bbox)
        else:
            comp_img = create_comparison_image(image, mask)
        comparison_images.append(comp_img)
        
        # DEBUG: 첫 2개 이미지를 파일로 저장하여 육안 확인
        if i < 2:
            try:
                import os
                debug_dir = "debug_verification"
                os.makedirs(debug_dir, exist_ok=True)
                save_path = os.path.join(debug_dir, f"triple_view_{i}.png")
                # RGB to BGR for cv2.imwrite
                comp_img_bgr = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, comp_img_bgr)
                if i == 0:
                    print(f"\n💾 DEBUG: Triple view saved to '{debug_dir}/' folder for inspection")
            except Exception as e:
                pass  # 저장 실패해도 계속 진행
    
    # Step 2: 모든 프롬프트 생성
    prompts = [create_verification_prompt(query, comparison_type) for query in queries]
    
    # Step 3: MLLM 배치 예측 (한 번에!)
    batch_responses = model.predict_batch(comparison_images, prompts)
    
    # DEBUG: 첫 번째 배치의 첫 2개 응답 상세 출력 + Parsing 결과
    if len(batch_responses) > 0:
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Verification Raw Responses & Parsing")
        print(f"{'='*80}")
        
        # 먼저 파싱 수행 (미리보기용)
        debug_parsed = []
        for i in range(min(2, len(batch_responses))):
            if isinstance(batch_responses[i], dict):
                text = batch_responses[i].get('reasoning', str(batch_responses[i]))
            else:
                text = str(batch_responses[i])
            debug_parsed.append(parse_verification_response(text))
        
        for i in range(min(2, len(batch_responses))):
            if isinstance(batch_responses[i], dict):
                text = batch_responses[i].get('reasoning', str(batch_responses[i]))
            else:
                text = str(batch_responses[i])
            
            print(f"\n[Sample {i+1}] Query: {queries[i][:70]}...")
            print(f"Raw Response:")
            print(text[:500] + ('...' if len(text) > 500 else ''))
            print(f"\nParsed:")
            print(f"  - Object: {debug_parsed[i]['identified_object'][:60]}...")
            print(f"  - Match: {debug_parsed[i]['matches']}")
            print(f"  - Reasoning: {debug_parsed[i]['reasoning'][:80]}...")
            print("-" * 80)
        print(f"{'='*80}\n")
    
    # Step 4: 모든 응답 파싱
    results = []
    for response in batch_responses:
        # 응답이 Dict가 아니라 str인 경우 처리
        if isinstance(response, dict):
            generated_text = response.get('reasoning', str(response))
        else:
            generated_text = str(response)
        
        # 파싱
        result = parse_verification_response(generated_text)
        results.append(result)
    
    return results


