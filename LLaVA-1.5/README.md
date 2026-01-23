# LLaVA-1.5 Jailbreak Attack

LLaVA-1.5-7B 모델을 대상으로 한 멀티모달 재일브레이크 공격 프로젝트입니다.

## 프로젝트 구조

```
LLaVA-1.5/
├── attack/              # 공격 코드
│   ├── text_attack.py          # 텍스트 공격 모듈
│   ├── visual_attack.py         # 이미지 공격 모듈
│   └── multimodal_step_jailbreak.py  # 멀티모달 공격 오케스트레이션
├── utils/               # 유틸리티
│   ├── model_loader.py         # LLaVA 모델 로더
│   ├── prompt_wrapper.py       # 프롬프트 래퍼
│   ├── generator.py            # 텍스트 생성기
│   └── data_utils.py           # 데이터 로딩 유틸리티
├── results/             # 결과 저장 디렉터리
│   └── adv_images/      # 공격 이미지 저장
├── config.py            # 설정 관리
├── main.py              # 메인 실행 스크립트
└── requirements.txt     # 의존성 목록
```

## 설정

1. **환경 변수 설정**
   ```bash
   cp .env.example .env
   # .env 파일을 편집하여 설정
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

## 실행

```bash
python main.py \
    --model_path llava-hf/llava-1.5-7b-hf \
    --batch_size 2 \
    --iters 50 \
    --n_train_data 520
```

### 주요 인자

- `--model_path`: LLaVA 모델 경로 (HuggingFace ID 또는 로컬 경로)
- `--batch_size`: 배치 크기 (기본값: 2, GPU 메모리에 따라 조정)
- `--iters`: 공격 반복 횟수 (기본값: 50)
- `--n_train_data`: 사용할 학습 데이터 수 (기본값: 520)
- `--load_in_8bit`: 8-bit 양자화 사용 (메모리 절약)

## 결과

- 공격 결과는 `results/{name}_results.json`에 저장됩니다.
- 10 epoch마다 모든 테스트 질문(470개)에 대해 평가하고 결과를 저장합니다.
- 공격 이미지는 `results/adv_images/`에 저장됩니다.
