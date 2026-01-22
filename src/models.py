"""
ReasonSeg Competition - Model Wrappers
4개 모델 래퍼: LLaVA, Qwen2.5-VL, Qwen3-VL, SAM2.1
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

from src.config import (
    MODEL_PATHS, GENERATION_CONFIG, GENERATION_CONFIG_SINGLE, GENERATION_CONFIG_MULTI,
    MODEL_LOAD_CONFIG, MODEL_SELECTION, MODEL_LOADING_STRATEGY, IMAGE_SIZE,
)
from src.prompts import (
    create_prompt, parse_response,
    fix_bbox_format
)
from src.multi import parse_multi_response


# Base Model Class

class BaseMLLM:
    """MLLM 모델의 기본 클래스"""
    
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self):
        """모델 로드 (서브클래스에서 구현)"""
        raise NotImplementedError
    
    def predict(self, image: np.ndarray, query: str) -> Dict:
        """
        예측 수행
        
        Args:
            image: (H, W, 3) RGB numpy array
            query: 텍스트 쿼리
        
        Returns:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'reasoning': str
            }
        """
        raise NotImplementedError
    
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """Numpy array를 PIL Image로 변환"""
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image


# LLaVA Model

class LLaVAModel(BaseMLLM):
    """
    LLaVA-1.5-7B 모델
    LISA 스타일 프롬프팅 사용
    """
    
    def __init__(self):
        super().__init__("LLaVA-1.5-7B", MODEL_PATHS['llava'])
    
    def load(self):
        """모델 로드"""
        print(f"Loading {self.model_name}...")
        
        # torch dtype 설정 (강의자료 방식: CUDA 가용성 체크)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=MODEL_LOAD_CONFIG['device_map'],
            attn_implementation=MODEL_LOAD_CONFIG['attn_implementation'],
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print(f"{self.model_name} loaded successfully")
    
    def predict(self, image: np.ndarray, query: str) -> Dict:
        """
        LLaVA로 Bbox 예측
        
        Args:
            image: (H, W, 3) RGB array
            query: 원본 쿼리
        
        Returns:
            prediction: {'bbox': [...], 'confidence': ..., 'reasoning': ...}
        """
        # LISA 스타일 프롬프트 생성
        prompt = create_lisa_style_prompt(query)
        
        # 이미지 준비
        pil_image = self._prepare_image(image)
        
        # 입력 준비
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=pil_image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)
        
        # 추론
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=GENERATION_CONFIG['do_sample'],
                max_new_tokens=GENERATION_CONFIG['max_new_tokens'],
                temperature=GENERATION_CONFIG['temperature'] if GENERATION_CONFIG['do_sample'] else None,
            )
        
        # 디코딩
        generated_text = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 응답 파싱
        result = parse_lisa_response(generated_text)
        
        # Bbox 검증 및 수정
        if result['bbox'] is not None:
            result['bbox'] = fix_bbox_format(result['bbox'], image.shape[:2][::-1])
        else:
            # 파싱 실패 시 기본 bbox
            result['bbox'] = fix_bbox_format(None, image.shape[:2][::-1])
            result['confidence'] = 0.3
        
        return result


# Qwen2.5-VL Model (LENS 스타일)

class Qwen25Model(BaseMLLM):
    """
    Qwen2.5-VL-7B 모델
    LENS 스타일 CoT 프롬프팅 사용
    """
    
    def __init__(self):
        super().__init__("Qwen2.5-VL-7B", MODEL_PATHS['qwen25'])
    
    def load(self):
        """모델 로드"""
        print(f"Loading {self.model_name}...")
        
        # torch dtype 설정
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=MODEL_LOAD_CONFIG['device_map'],
            attn_implementation=MODEL_LOAD_CONFIG['attn_implementation'],
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print(f"{self.model_name} loaded successfully")
    
    def predict(self, image: np.ndarray, query: str) -> Dict:
        """
        Qwen2.5-VL로 Bbox 예측 (CoT)
        
        Args:
            image: (H, W, 3) RGB array
            query: 원본 쿼리
        
        Returns:
            prediction: {'bbox': [...], 'confidence': ..., 'reasoning': ...}
        """
        # LENS 스타일 CoT 프롬프트
        prompt = create_lens_style_prompt(query)
        
        # 이미지 준비
        pil_image = self._prepare_image(image)
        
        # Qwen2.5-VL 입력 형식
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        ).to(self.device)
        
        # 추론
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=GENERATION_CONFIG['do_sample'],
                max_new_tokens=GENERATION_CONFIG['max_new_tokens'],
            )
        
        # 디코딩
        generated_text = self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # 응답 파싱
        result = parse_lens_response(generated_text)
        
        # Bbox 검증
        if result['bbox'] is not None:
            result['bbox'] = fix_bbox_format(result['bbox'], image.shape[:2][::-1])
        else:
            result['bbox'] = fix_bbox_format(None, image.shape[:2][::-1])
            result['confidence'] = 0.3
        
        return result


# 3. Qwen3-VL Model

class Qwen3Model(BaseMLLM):
    """
    Qwen3-VL-8B 모델
    Spatial Reasoning 프롬프팅
    """
    
    def __init__(self):
        super().__init__("Qwen3-VL-8B", MODEL_PATHS['qwen3'])
    
    def load(self):
        """모델 로드"""
        print(f"Loading {self.model_name}...")
        
        # torch dtype 설정
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=MODEL_LOAD_CONFIG['device_map'],
            attn_implementation=MODEL_LOAD_CONFIG['attn_implementation'],
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print(f"{self.model_name} loaded successfully")
    
    def _get_model_device(self):
        """device_map='auto'일 때 실제 모델 device 반환"""
        if self.model is None:
            return self.device
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device
    
    def _encode_qwen3(self, pil_image: Image.Image, prompt: str) -> dict:
        """Qwen3-VL 공식 vision 전처리 (process_vision_info + do_resize=False)"""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(
            messages, image_patch_size=16
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        return inputs.to(self._get_model_device())
    
    def _encode_qwen3_batch(self, pil_images: List[Image.Image], prompts: List[str]) -> dict:
        """배치용 Qwen3-VL 입력 (샘플별 process_vision_info)"""
        from qwen_vl_utils import process_vision_info

        texts = []
        batch_images = []

        for pil_image, prompt in zip(pil_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages, image_patch_size=16)
            texts.append(text)
            if image_inputs:
                batch_images.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=batch_images if batch_images else None,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        return inputs.to(self._get_model_device())
    
    def _generate_text(self, inputs: dict, gen_config: dict) -> str:
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_config)
        return self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    
    def predict(
        self,
        image: np.ndarray,
        query: str,
        feedback: str = None,
        is_multi: bool = None,
    ) -> Dict:
        """
        Qwen3-VL로 Bbox 예측
        
        Args:
            image: (H, W, 3) RGB array
            query: 원본 쿼리
            feedback: v1 refinement 피드백 (optional)
            is_multi: None → v1/v2 (active_prompt)
                      False → v3 single (active_single)
                      True  → v3 multi  (active_multi)
        
        Returns:
            prediction dict (single: bbox / multi: bboxes)
        """
        pil_image = self._prepare_image(image)

        if is_multi is True:
            prompt = create_prompt(query, mode='multi')
            gen_config = GENERATION_CONFIG_MULTI
            inputs = self._encode_qwen3(pil_image, prompt)
            generated_text = self._generate_text(inputs, gen_config)
            result = parse_multi_response(generated_text)
            result['raw_response'] = generated_text
            result['is_multi'] = True
            return result

        if is_multi is False:
            prompt = create_prompt(query, mode='single')
            gen_config = GENERATION_CONFIG_SINGLE
            bbox_size = IMAGE_SIZE
        else:
            prompt = create_prompt(query, feedback)
            gen_config = GENERATION_CONFIG
            bbox_size = image.shape[:2][::-1]

        inputs = self._encode_qwen3(pil_image, prompt)
        generated_text = self._generate_text(inputs, gen_config)
        result = parse_response(generated_text)

        if result['bbox'] is not None:
            result['bbox'] = fix_bbox_format(result['bbox'], bbox_size)
        else:
            result['bbox'] = fix_bbox_format(None, bbox_size)
            result['confidence'] = 0.3

        result['raw_response'] = generated_text
        result['is_multi'] = is_multi
        return result
    
    def predict_batch(self, images: List[np.ndarray], queries: List[str], feedbacks: List[str] = None) -> List[Dict]:
        """
        배치로 여러 이미지 동시 처리
        
        Args:
            images: 이미지 리스트 [(H, W, 3), ...]
            queries: 쿼리 리스트 [str, ...]
            feedbacks: 피드백 리스트 [str or None, ...] (optional)
        
        Returns:
            predictions: [{'bbox': ..., 'confidence': ..., 'reasoning': ...}, ...]
        """
        # PIL 변환
        pil_images = [self._prepare_image(img) for img in images]
        
        # 프롬프트 생성 (feedback 포함)
        if feedbacks is None:
            feedbacks = [None] * len(queries)
        prompts = [create_prompt(q, f) for q, f in zip(queries, feedbacks)]
        
        inputs = self._encode_qwen3_batch(pil_images, prompts)
        
        # 배치 추론
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **GENERATION_CONFIG,
            )
        
        # 배치 디코딩
        generated_texts = self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # 각 응답 파싱
        results = []
        for i, (generated_text, image) in enumerate(zip(generated_texts, images)):
            result = parse_response(generated_text)
            
            # Bbox 검증
            if result['bbox'] is not None:
                result['bbox'] = fix_bbox_format(result['bbox'], image.shape[:2][::-1])
            else:
                result['bbox'] = fix_bbox_format(None, image.shape[:2][::-1])
                result['confidence'] = 0.3
            
            results.append(result)
        
        return results


# SAM2.1 Model

class SAMModel:
    """
    SAM2.1 Segmentation 모델
    """
    
    def __init__(self):
        self.model_name = "SAM2.1"
        self.weights_path = MODEL_PATHS['sam_weights']
        self.config_path = MODEL_PATHS['sam_config']
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self):
        """SAM2.1 로드"""
        print(f"Loading {self.model_name}...")
        
        sam_model = build_sam2(self.config_path, self.weights_path)
        self.predictor = SAM2ImagePredictor(sam_model)
        
        print(f"{self.model_name} loaded successfully")
    
    def set_image(self, image: np.ndarray):
        """
        이미지 설정 (embedding 생성)
        
        Args:
            image: (H, W, 3) RGB numpy array
        """
        self.predictor.set_image(image)
    
    def predict_with_bbox(self, bbox: List[float]) -> Tuple[np.ndarray, float]:
        """
        Bbox 프롬프트로 마스크 예측
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            mask: (H, W) bool array
            score: mask quality score
        """
        bbox_np = np.array(bbox)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_np[None, :],  # (1, 4)
            multimask_output=True,
        )
        
        # 가장 높은 score의 mask 선택
        best_idx = scores.argmax()
        return masks[best_idx].astype(bool), scores[best_idx]
    
    def predict_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Point 프롬프트로 마스크 예측
        
        Args:
            point_coords: (N, 2) [[x, y], ...] 
            point_labels: (N,) [1, 1, ..., 0, 0] (1=positive, 0=negative)
        
        Returns:
            mask: (H, W) bool array
            score: mask quality score
        """
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=False,  # Single mask
        )
        
        return masks[0].astype(bool), scores[0]


# Model Manager (모든 모델 통합 관리)

class ModelManager:
    """모델 관리자"""
    
    def __init__(self):
        self.llava = None
        self.qwen25 = None
        self.qwen3 = None
        self.sam = SAMModel()
        
        # 활성화된 모델 목록
        self.active_models = []
        if MODEL_SELECTION['use_llava']:
            self.active_models.append('llava')
        if MODEL_SELECTION['use_qwen25']:
            self.active_models.append('qwen25')
        if MODEL_SELECTION['use_qwen3']:
            self.active_models.append('qwen3')
        
        if len(self.active_models) == 0:
            raise ValueError("최소 1개 이상의 MLLM을 활성화해야 합니다!")
        
        print(f"Active models: {', '.join(self.active_models)}")
        print(f"Loading strategy: {MODEL_LOADING_STRATEGY}")
    
    def load_sam(self):
        """SAM만 먼저 로드"""
        print("Loading SAM2.1...")
        self.sam.load()
        print("SAM2.1 loaded")
    
    
    def unload_model(self, model_name: str):
        """모델 언로드 및 메모리 정리"""
        if model_name == 'llava' and self.llava is not None:
            del self.llava.model
            del self.llava.processor
            self.llava = None
        elif model_name == 'qwen25' and self.qwen25 is not None:
            del self.qwen25.model
            del self.qwen25.processor
            self.qwen25 = None
        elif model_name == 'qwen3' and self.qwen3 is not None:
            del self.qwen3.model
            del self.qwen3.processor
            self.qwen3 = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

