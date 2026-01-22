"""
ReasonSeg Competition - Prompt Templates
Qwen3-VL용 프롬프트 (YAML 기반)
"""

import re
from pathlib import Path
from typing import Any
from typing import Dict, List, Tuple


# 프롬프트 생성

def create_prompt(query: str, feedback: str = None, mode: str = None) -> str:
    """
    prompts.yaml 기반 프롬프트 생성

    Args:
        query: 원본 쿼리
        feedback: 이전 실패 시 피드백 (optional, v1 refinement)
        mode: None → active_prompt (v1/v2)
              'single' → active_single (v3/v4 hybrid)
              'multi'  → active_multi  (v3/v4 hybrid)

    Returns:
        formatted_prompt: 모델 입력용 프롬프트
    """
    template = _load_prompt_template(mode)
    
    prompt = template.replace('{query}', query)
    
    # Feedback이 있으면 프롬프트에 추가
    if feedback:
        feedback_section = f"\n\n{'='*60}\nIMPORTANT - Previous Attempt Feedback:\n{'='*60}\n{feedback}\n{'='*60}\n\nBased on this feedback, provide a DIFFERENT bounding box.\n"
        # "### Your Task ###" 뒤에 feedback 삽입
        prompt = prompt.replace('### Your Task ###', '### Your Task ###' + feedback_section)
    
    return prompt


_PROMPT_CACHE: dict[str, Any] | None = None


def _load_prompt_template(mode: str = None) -> str:
    """
    prompts.yaml에서 템플릿 로드

    - mode=None: active_prompt (v1/v2)
    - mode='single'|'multi': active_single / active_multi (v3 hybrid)
    """
    global _PROMPT_CACHE

    if _PROMPT_CACHE is None:
        yaml_path = Path(__file__).with_name("prompts.yaml")
        if not yaml_path.exists():
            yaml_path = Path(__file__).parent.parent / "prompts.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"prompts.yaml not found in src/ or project root: {yaml_path}"
            )

        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError(
                "PyYAML이 필요합니다. 서버에서 `pip install pyyaml` 후 다시 실행하세요."
            ) from e

        with yaml_path.open("r", encoding="utf-8") as f:
            _PROMPT_CACHE = yaml.safe_load(f)

    prompts = _PROMPT_CACHE.get("prompts", {})

    if mode == 'single':
        active = _PROMPT_CACHE.get("active_single")
    elif mode == 'multi':
        active = _PROMPT_CACHE.get("active_multi")
    else:
        active = _PROMPT_CACHE.get("active_prompt")

    template = prompts.get(active)

    if not active or not template:
        raise ValueError(
            f"Invalid prompts.yaml: prompt key for mode={mode!r} not found (active='{active}')."
        )

    if "{query}" not in template:
        raise ValueError(
            f"Prompt template '{active}' must include '{{query}}' placeholder."
        )

    return template


def parse_response(response: str) -> Dict:
    """
    응답 파싱
    
    작동하는 예제 파일(new_qwen3b)의 방식을 참고하여 BBox 추출 강화
    
    Args:
        response: 모델 출력 텍스트
    
    Returns:
        parsed: {
            'bbox': [x1, y1, x2, y2],
            'confidence': float,
            'reasoning': str
        }
    """
    result = {
        'bbox': None,
        'confidence': 0.5,
        'reasoning': response  # 전체 응답을 reasoning으로 저장
    }
    
    # 방법 1: BBox: [x1, y1, x2, y2] 형식
    bbox_match = re.search(r'BBox:\s*\[([^\]]+)\]', response, re.IGNORECASE)
    if bbox_match:
        try:
            coords = [float(x.strip()) for x in bbox_match.group(1).split(',')]
            if len(coords) == 4:
                result['bbox'] = coords
        except:
            pass
    
    # 방법 2: Bounding Box: [x1, y1, x2, y2] 형식
    if result['bbox'] is None:
        bbox_match = re.search(r'Bounding Box:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        if bbox_match:
            try:
                coords = [float(x.strip()) for x in bbox_match.group(1).split(',')]
                if len(coords) == 4:
                    result['bbox'] = coords
            except:
                pass
    
    # 방법 3: [x1, y1, x2, y2] 형식 (BBox: 없이, 정규표현식 강화)
    if result['bbox'] is None:
        json_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
        match = re.search(json_pattern, response)
        if match:
            try:
                result['bbox'] = [float(match.group(i)) for i in range(1, 5)]
            except:
                pass
    
    # Confidence 추출 (여러 패턴 시도)
    conf_match = re.search(r'Confidence:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE)
    if conf_match:
        try:
            result['confidence'] = float(conf_match.group(1))
        except:
            pass
    
    return result


# 공통 유틸리티

def fix_bbox_format(bbox: List[float], image_size: Tuple[int, int] = (1000, 1000)) -> List[float]:
    """
    Bbox 형식 수정 (범위 초과 시 클리핑)
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_size: (width, height)
    
    Returns:
        fixed_bbox: [x1, y1, x2, y2]
    """
    if bbox is None:
        # 기본 bbox 반환 (이미지 중앙 1/4)
        w, h = image_size
        return [w*0.375, h*0.375, w*0.625, h*0.625]
    
    x1, y1, x2, y2 = bbox
    w, h = image_size
    
    # 클리핑
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h))
    
    # 좌표 순서 수정
    if x2 <= x1:
        x1, x2 = x2, x1
    if y2 <= y1:
        y1, y2 = y2, y1
    
    # 최소 크기 보장
    if (x2 - x1) < 10:
        center_x = (x1 + x2) / 2
        x1 = max(0, center_x - 5)
        x2 = min(w, center_x + 5)
    
    if (y2 - y1) < 10:
        center_y = (y1 + y2) / 2
        y1 = max(0, center_y - 5)
        y2 = min(h, center_y + 5)
    
    return [x1, y1, x2, y2]
