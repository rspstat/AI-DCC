# Mission 2 — 신고자 / 119대원 화자 분류

119 응급 통화의 발화 구간을 **음성만으로** 듣고 신고자(1)인지 119대원(0)인지 분류합니다.
(전사 텍스트와 발화 순서 정보는 모델 입력으로 사용하지 않습니다.)

## 1. 폴더 구성

```
.
├── inference.py                 # 채점 시 실행되는 파일
├── model.py                     # 모델 구조 (학습 코드와 동일한 구조)
├── requirements.txt             # 패키지 버전 고정
├── train_m2.ipynb               # 학습 과정 기록 (셀 출력 포함)
├── export_wavlm_config.py       # 1회성 준비 스크립트 (채점 시 실행되지 않음)
├── pretrained_config/
│   └── wavlm-base/config.json   # WavLM 구조 설정 (가중치 아님, 오프라인 로드용)
└── checkpoints/
    └── best_model_wavlm_mission2_v9_12layers.pt   # 학습된 가중치
```

## 2. 실행 방법

```bash
pip install -r requirements.txt

python inference.py \
    --audio_dir {wav 폴더} \
    --label_dir {json 폴더} \
    --ckpt_path checkpoints/best_model_wavlm_mission2_v9_12layers.pt \
    --output ./outputs/mission2.csv
```

- 위 명령어 **1회 실행**으로 결과 CSV가 생성됩니다. `./outputs/` 폴더가 없으면 자동으로 만듭니다.
- GPU가 있으면 GPU, 없으면 CPU로 자동 전환됩니다.
- **인터넷 없이 동작**합니다. 사전학습 가중치를 Hub에서 내려받지 않고, `pretrained_config/wavlm-base/config.json`(구조 설정)만 읽은 뒤 `--ckpt_path`의 체크포인트로 모든 가중치를 덮어씁니다.
- 출력 CSV 컬럼: `audio file name, startAt, endAt, speaker` (speaker: 0=119대원, 1=신고자)
- 분류 기준: 2클래스 softmax에서 확률이 큰 쪽을 선택합니다(argmax). 이는 **threshold 0.5 고정**과 같으며, 클래스별 threshold 조정은 하지 않습니다.
- 입력으로는 json의 `startAt`, `endAt`(시간 구간)만 읽고 `text`, `speaker` 등은 읽지 않습니다.
- 모든 발화에 같은 처리를 적용합니다. 학습 때는 300ms 미만 발화를 제외했지만, 추론에서는 제출 CSV에 모든 발화의 예측이 필요하므로 제외하지 않습니다.

## 3. 모델 요약

| 항목 | 내용 |
|---|---|
| 구조 | WavLM-base 12층(전체 fine-tuning) → Attention Pooling → Dropout(0.3) → Linear(2) |
| 입력 | 16kHz, 2.5초 고정 길이 파형 (짧으면 무음 패딩, 길면 앞부분 사용), 발화별 z-score 정규화 |
| 학습 | Adam(weight_decay 1e-4), 기준 lr 1e-5에서 아래 층일수록 ×0.9 감쇠, ReduceLROnPlateau, CrossEntropy, batch 32 |
| 학습 데이터 | 대회 Training(서울) 29,200개 통화, 통화 단위 8:2 분할(seed 42) |
| 사전학습 모델 | `microsoft/wavlm-base` (오픈소스 pre-trained 모델) |

학습 상세와 로그는 `mission2.ipynb`에 있습니다.

## 4. 계산 효율 정보

| 항목 | 값 |
|---|---|
| Total 파라미터 수 | 94,384,243 |
| Active 파라미터 수 | 85,462,771 (90.55%, 학습 때 fine-tuning한 부분) |
| 추론 환경 | NVIDIA GeForce RTX 3080 10GB (CPU: Intel i7-13700K, RAM 128GB), `num_workers=0` |
| Validation 추론 batch size | 16 |
| Validation 전체 추론시간 | 336.40초 (111,947개 발화) |
| 샘플당 평균 추론시간 | 약 3.00 ms |
| 학습 시간 | 약 20.2시간 (15 epoch, batch 32) |

`inference.py` 실행 시 위 값(환경, batch size, 파라미터 수, 추론시간)이 화면에 출력됩니다.

## 5. 성능

| 평가 | Accuracy |
|---|---|
| 자체 분할 Val (8:2, 176,784 발화) | 93.00% |
| 대회 공식 Validation (3,640통화) | 92.64% |
| 공식 Validation, 개인정보 비식별화 구간 제외 | 97.07% |

체크포인트는 자체 분할 Val Loss 기준으로 선택했으며, 공식 Validation은 학습과 체크포인트 선택에 사용하지 않았습니다.
