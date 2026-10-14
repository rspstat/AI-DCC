# Mission 1. 신고자 성별 분류

119 신고 통화 음성에서 신고자의 성별(M/F)을 분류합니다.

## 실행 방법

```bash
pip install -r requirements.txt

python inference.py \
    --audio_dir {wav 폴더} \
    --label_dir {json 폴더} \
    --ckpt_path weights/clf.joblib \
    --output ./outputs/mission1.csv
```

- 명령어 1회 실행으로 결과 CSV가 생성됩니다.
- `--ckpt_path`에는 `weights/clf.joblib`을 지정합니다. ECAPA-TDNN 가중치는 같은 폴더의 `weights/ecapa/`에서 자동으로 불러옵니다.
- 사전학습 가중치를 모두 포함하고 있어 **인터넷 연결 없이 실행**됩니다. (`HF_HUB_OFFLINE=1` 설정)
- GPU가 있으면 GPU, 없으면 CPU로 실행됩니다.
- 출력 폴더가 없으면 자동으로 생성합니다.

## 출력 형식

```
audio file name,gender
651e464d69a4f266f0626837_20220101.wav,M
651e464d69a4f266f0626846_20220101.wav,F
```

## 폴더 구조

```
mission1/
├── inference.py          # 추론 실행 스크립트 (진입점)
├── model.py              # 모델 정의 및 전처리 (GenderClassifier)
├── train_m1.ipynb      # 학습 코드 및 학습 로그
├── requirements.txt
├── README.md
└── weights/
    ├── clf.joblib        # 학습한 분류기 (StandardScaler + LogisticRegression)
    └── ecapa/            # 사전학습 ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb)
        ├── hyperparams.yaml
        ├── embedding_model.ckpt
        ├── mean_var_norm_emb.ckpt
        ├── classifier.ckpt       # 모델 로딩 시 필요 (추론에는 미사용)
        └── label_encoder.txt    # 모델 로딩 시 필요 (추론에는 미사용)
```

`weights/ecapa/` 폴더는 `clf.joblib`과 같은 위치에 있어야 합니다.

## 모델 구조

| 단계 | 내용 |
|---|---|
| 입력 | 오디오 + 라벨링데이터의 `startAt`, `endAt`, `speaker`만 사용 |
| 발화 추출 | 신고자(speaker=1) 발화 구간을 잘라 8kHz → 16kHz 리샘플링, 0.3초 미만 발화 제외 |
| 특징 추출 | ECAPA-TDNN(가중치 고정)으로 발화별 192차원 임베딩 추출 후 L2 정규화 |
| 통화 벡터 | 발화 길이로 가중 평균해 통화당 벡터 1개 생성 |
| 분류 | StandardScaler + LogisticRegression, **threshold 0.5 고정** |

## 성능 (AI-hub Validation, 서울 3,640통화)

| Accuracy | F1 (F 기준) |
|---|---|
| 0.9821 | 0.9834 |

- 학습에는 Training(서울) 29,200통화만 사용했습니다.
- Validation은 집계 방식 선정과 성능 확인에만 사용했으며, 학습(역전파)에는 사용하지 않았습니다.
- seed: 42

## 계산 효율 정보

| 항목 | 값 |
|---|---|
| Total 파라미터 수 | 22,151,105 (ECAPA 패키지 전체 22,150,912 + 로지스틱 회귀 193) |
| Active 파라미터 수 | 20,767,745 (ECAPA 임베딩 모델 20,767,552 + 로지스틱 회귀 193, 93.75%) |
| 컴퓨팅 환경 | Google Colab, NVIDIA Tesla T4 (15GB) |
| Validation 추론 batch size | 통화 단위 (한 통화의 신고자 발화 전체를 padding해 1 batch로 처리) |
| Validation 전체 추론시간 | 664.5초 (3,640통화) |
| 샘플당 평균 추론시간 | 182.5ms / 통화 |
