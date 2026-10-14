# Mission 3. 환자 증상 인식

119 신고 통화의 대화 전사 텍스트로 환자 증상 9종(멀티라벨)을 인식합니다.

## 실행 방법

```bash
pip install -r requirements.txt

python inference.py \
    --audio_dir {wav 폴더} \
    --label_dir {json 폴더} \
    --ckpt_path ckpt_m3 \
    --output ./outputs/mission3.csv
```

- 명령어 1회 실행으로 결과 CSV가 생성됩니다.
- `--ckpt_path`에는 체크포인트 **폴더** `ckpt_m3`를 지정합니다. (`klue/`, `kcelectra/` 하위 폴더 포함)
- `--audio_dir`은 대회 실행 형식을 맞추기 위한 인자이며, Mission 3는 텍스트만 사용하므로 읽지 않습니다.
- 토크나이저·config·가중치를 모두 포함하고 있어 **인터넷 연결 없이 실행**됩니다. (`HF_HUB_OFFLINE=1` 설정)
- GPU가 있으면 GPU, 없으면 CPU로 실행됩니다.

## 출력 형식

```
label file name,symptom
651e464d69a4f266f0626837_20220101.json,"[""구토"", ""오심""]"
651e464d69a4f266f0626845_20220101.json,[]
```

- `symptom`은 라벨 JSON과 같은 리스트 문자열입니다. 예: `["구토", "오심"]`, 증상이 없으면 `[]`
- CSV 안의 `""`는 따옴표를 표현하는 표준 CSV 처리입니다.

## 폴더 구조

```
mission3/
├── inference.py                     # 추론 실행 스크립트 (진입점)
├── model_defs.py                    # 라벨 순서, 입력 텍스트 구성, KLUE-RoBERTa 모델 구조
├── mission3_train_klue_roberta.ipynb    # KLUE-RoBERTa 학습 코드 및 로그
├── Mission3_train_kcelectra.ipynb              # KcELECTRA 학습 코드 및 로그
├── requirements.txt
├── README.md
└── ckpt_m3/
    ├── klue/        # config, 토크나이저, seed42.pt, seed123.pt, seed2026.pt
    └── kcelectra/   # config, 토크나이저, seed42.pt, seed123.pt, seed2026.pt
```

## 모델 구조

| 항목 | KLUE-RoBERTa | KcELECTRA |
|---|---|---|
| 사전학습 모델 | `klue/roberta-base` | `beomi/KcELECTRA-base-v2022` |
| 분류 head | mean pooling + MLP(768→256→9) | HF 기본 분류 head |
| Loss | BCEWithLogits + pos_weight (Train 음성/양성 비율) | BCEWithLogits (pos_weight 없음) |
| 학습 설정 | AdamW (backbone 2e-5, head 1e-3), warmup 10%, epoch 상한 8, patience 3 | AdamW 3e-5, warmup 6%, epoch 상한 5, patience 2 |
| 공통 | max_length 512, batch 16, fp16, seed 42 / 123 / 2026 | |

- **입력**: 라벨 JSON의 `utterances`를 `startAt` 순으로 정렬해 `text`만 공백으로 연결 (화자 태그·다른 필드 미사용)
- **앙상블** (Validation으로 가중치를 고르지 않는 고정 규칙)
  1. 백본별로 seed 3개의 확률 평균
  2. 두 백본 확률을 1:1 평균
  3. 모든 클래스 **threshold 0.5 고정**

## 성능 (AI-hub Validation, 서울 3,640건, Macro F1)

| 모델 | Macro F1 |
|---|---|
| KLUE-RoBERTa (3 seed 평균) | 0.6246 |
| KcELECTRA (3 seed 평균) | 0.6136 |
| **최종 앙상블** | **0.6543** |

- 학습에는 Training(서울) 29,200건만 사용했습니다.
- Validation은 best epoch 선정과 앙상블 조합 선정에만 사용했으며, 학습(역전파)에는 사용하지 않았습니다.
- 제출용 `inference.py` 결과가 학습 시 저장한 확률의 앙상블과 일치함을 확인했습니다. (판정이 바뀐 샘플·클래스 0 / 32,760)

## 계산 효율 정보

| 항목 | 값 |
|---|---|
| Total 파라미터 수 | 약 715M (KLUE-RoBERTa 약 110.6M × 3 + KcELECTRA 약 127.8M × 3) |
| Active 파라미터 수 | 약 715M (추론 시 6개 모델 모두 사용) |
| 컴퓨팅 환경 | Google Colab, NVIDIA Tesla T4 (15GB) |
| Validation 추론 batch size | 32 |
| Validation 전체 추론시간 | 167.5초 (3,640건, 모델 6개 순차 실행, 모델 로딩 포함) |
| 샘플당 평균 추론시간 | 46ms |
