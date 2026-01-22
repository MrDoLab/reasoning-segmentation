"""
ReasonSeg Competition - Configuration
모든 경로, 설정값, 하이퍼파라미터를 관리
"""

# 기본 설정

SEED = 428


# 경로 설정

# 데이터셋 경로
DATASET_DIR = "/home/jjs2403/2026_bootcamp_02/dataset/challenge_datasets/challenge1"
TEST_CSV = f"{DATASET_DIR}/query.csv"

# 모델 경로
MODEL_BASE_DIR = "/home/jjs2403/2026_bootcamp_02/models"

MODEL_PATHS = {
    # MLLM 모델들
    'llava': f"{MODEL_BASE_DIR}/llava-v1.5-7b",
    'qwen25': f"{MODEL_BASE_DIR}/Qwen2.5-VL-7B-Instruct",
    'qwen3': f"{MODEL_BASE_DIR}/Qwen3-VL-8B-Instruct",
    
    # SAM2.1 (강의자료 형식 사용)
    'sam_weights': "//home//jjs2403//2026_bootcamp_02//models//SAM2.1//weights//sam2.1_hiera_large.pt",
    'sam_config': "//home//jjs2403//2026_bootcamp_02//models//SAM2.1//weights//sam2.1_hiera_l.yaml",
}

# 모델 설정

# 모델 선택 (True/False로 각 모델 on/off)
MODEL_SELECTION = {
    'use_llava': False,
    'use_qwen25': False,
    'use_qwen3': True,
}

# 모델 로딩 전략
MODEL_LOADING_STRATEGY = 'parallel'  # 'sequential' (순차) or 'parallel' (2개씩 동시)

# 추론 설정 
GENERATION_CONFIG = {
    'do_sample': False,        # 필수: 결정적 추론
    'max_new_tokens': 256,     # 128→256 (BBox까지 생성되도록 충분한 길이)
    'temperature': 0.0,
    'top_p': None,
    'top_k': None,
}

# 모델 로딩 설정
MODEL_LOAD_CONFIG = {
    'torch_dtype': 'bfloat16',  # or 'float16'
    'device_map': 'auto',
    'attn_implementation': 'flash_attention_2',
}

# 파이프라인 설정

# Iterative Refinement 설정
REFINEMENT_CONFIG = {
    'enabled': True,                 # Refinement 사용 여부 (기본 False)
    'max_rounds': 3,                  # 최대 반복 횟수
    'base_seed': 428,                 # 기본 Seed (각 라운드마다 +1씩 증가)
    'comparison_type': 'triple',      # 'side_by_side' or 'triple'
    'early_stop': True,               # 검증 통과 시 조기 종료
}

# Voting 설정 (v2)
VOTING_CONFIG = {
    'strategy': 'weighted_average',  # 'weighted_average', 'highest_conf', 'consensus'
    'min_confidence': 0.3,           # 최소 confidence threshold
    'iou_threshold': 0.5,            # Consensus voting용
}

# Hybrid 설정 (v3/v4) — final unified Cell 2
HYBRID_CONFIG = {
    'multi_keywords': ['all', 'every', 'both', 'each', 'segment all'],
    'keyword_match': 'word_boundary',
}

# v3 hybrid: single vs multi generation
GENERATION_CONFIG_SINGLE = {
    'do_sample': False,
    'max_new_tokens': 256,
    'temperature': 0.0,
    'top_p': None,
    'top_k': None,
}

GENERATION_CONFIG_MULTI = {
    'do_sample': False,
    'max_new_tokens': 1024,
    'temperature': 0.0,
    'top_p': None,
    'top_k': None,
}

# Final submission 설정 (v4) — frozen, 제출 시 이 값만 사용
FINAL_CONFIG = {
    'seed': 428,
    'verbose': True,
    'unload_mllm': True,
    'save_intermediate': False,
    'visualize': False,
    'output_dir': 'outputs',
    'submission_file': 'submission.csv',
}


# 이미지 설정
IMAGE_SIZE = (1000, 1000)


# 출력 설정

OUTPUT_DIR = "outputs"
SUBMISSION_FILE = "submission.csv"

# 중간 결과 저장 
SAVE_INTERMEDIATE = False            # True로 설정 시 중간 결과 저장
INTERMEDIATE_DIRS = {
    'stage1': f"{OUTPUT_DIR}/stage1_predictions",
    'stage2_voting': f"{OUTPUT_DIR}/stage2_voting",
    'stage2_sam': f"{OUTPUT_DIR}/stage2_sam",
    'stage3_sam': f"{OUTPUT_DIR}/stage3_sam",
    'visualizations': f"{OUTPUT_DIR}/visualizations",
}

# 디버깅 설정

DEBUG = False                        # 디버그 모드
VERBOSE = True                       # 진행 상황 출력
VISUALIZE = False                    # 시각화 생성 여부
