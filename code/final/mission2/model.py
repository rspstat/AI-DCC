"""
Mission 2 - WavLM 기반 화자(신고자/119대원) 분류 모델 정의

학습 코드(.ipynb)에서 사용한 것과 동일한 모델 구조입니다.
inference.py가 이 파일의 WavLMClassifierV2를 생성한 뒤,
학습이 끝난 체크포인트(.pt)의 가중치를 통째로 덮어씌워서 사용합니다.

※ 채점 환경이 인터넷에 연결되지 않을 수 있으므로, 사전학습 가중치를
  Hugging Face Hub에서 직접 내려받지 않습니다. 대신 미리 로컬에 저장해
  둔 설정(config.json)만 읽어서 "빈 구조"를 만들고, 실제 가중치는
  inference.py가 체크포인트로 전부 덮어씁니다. (로컬 설정 파일을
  준비하는 방법은 export_wavlm_config.py 참고 — 그 스크립트는 1회성
  준비 작업이며 채점 시에는 실행되지 않습니다)

※ 이 파일은 추론 전용입니다. 학습 코드에 있던 층별 freeze/unfreeze
  (requires_grad 설정, feature_extractor 고정 등) 로직은 일부러 뺐습니다.
  추론은 역전파(backward)를 하지 않으므로 requires_grad 값이 결과에
  전혀 영향을 주지 않기 때문입니다.
"""
import os                                         # 로컬 config 경로를 조합하기 위해 사용
import torch                                       # 텐서 연산(softmax 등)에 사용
import torch.nn as nn                               # 신경망 레이어(Linear, Dropout 등) 정의에 사용
from transformers import WavLMModel, WavLMConfig     # WavLM 모델 본체와, 가중치 없이 구조만 담는 설정 클래스

# 이 .py 파일이 실제로 위치한 폴더를 기준으로 로컬 config 경로를 계산
# (채점 서버가 어느 디렉토리에서 inference.py를 실행하든 항상 올바른 경로를 찾기 위함)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                        # 이 파일이 있는 폴더의 절대경로
_LOCAL_CONFIG_DIR = os.path.join(_THIS_DIR, "pretrained_config", "wavlm-base")  # 로컬에 저장해 둔 config 폴더 경로

if not os.path.isdir(_LOCAL_CONFIG_DIR):                                       # 로컬 config 폴더가 없으면
    raise FileNotFoundError(                                                     # 바로 알아챌 수 있게 명확한 에러를 냄
        f"로컬 WavLM 설정 폴더를 찾을 수 없습니다: {_LOCAL_CONFIG_DIR}\n"
        "export_wavlm_config.py를 먼저 1회 실행해 설정 파일을 준비한 뒤, "
        "그 결과 폴더(pretrained_config)를 이 model.py와 같은 위치에 포함시켜 주세요."
    )


class AttentionPooling(nn.Module):
    """
    시간(프레임) 축으로 요약할 때, 모든 프레임을 동일한 비중으로 평균 내는
    대신 프레임별 중요도(weight)를 학습해서 가중 평균하는 층.

    단순 평균(Mean Pooling) 대신 사용한 이유: 패딩으로 채워진 무음 구간은
    거의 0에 가까운 weight를 받아 결과에 영향을 주지 않고, 실제 발화가
    시작되는 지점에 높은 weight가 실린다는 것을 학습 단계의 Attention Map
    시각화로 직접 확인했습니다.
    """

    def __init__(self, hidden_size):                       # hidden_size: WavLM 출력 벡터의 차원 수
        super().__init__()                                   # nn.Module 초기화
        self.attention = nn.Linear(hidden_size, 1)            # 프레임 벡터 1개 -> 중요도 점수 1개로 매핑하는 선형층

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size) — WavLM 인코더의 프레임별 출력
        scores = self.attention(hidden_states)          # (batch, seq_len, 1) 프레임별 중요도 점수 계산
        weights = torch.softmax(scores, dim=1)            # seq_len(시간) 축 기준 정규화 -> 합이 1인 가중치로 변환
        pooled = (hidden_states * weights).sum(dim=1)      # 가중치를 곱한 뒤 시간축으로 합산 -> (batch, hidden_size)
        return pooled                                        # 발화 1개당 벡터 1개로 요약된 결과 반환


class WavLMClassifierV2(nn.Module):
    """
    WavLM 인코더 + Attention Pooling + 분류기(0=대원, 1=신고자)

    forward(input_values) 처리 흐름:
        (batch, 오디오 샘플 수) 파형
        -> WavLM 인코더 -> (batch, seq_len, hidden_size) 프레임별 벡터
        -> Attention Pooling -> (batch, hidden_size) 발화 1개당 벡터 1개
        -> 분류기 -> (batch, 2) 대원/신고자 logit

    num_layers_to_keep : WavLM 인코더에 유지할 층 수.
                          WavLM-base는 원래 12층이며, 최종 채택 모델(12층
                          전체 fine-tuning)은 12를 그대로 사용합니다.
                          체크포인트와 층 수가 다르면 load_state_dict가
                          실패하므로, 학습 때와 반드시 동일한 값을 써야 합니다.
    """

    def __init__(self, num_classes=2, num_layers_to_keep=12):   # num_classes: 분류할 클래스 수(대원/신고자=2)
        super().__init__()                                        # nn.Module 초기화

        # 인터넷에서 가중치를 내려받지 않고, 로컬에 저장해 둔 설정(config)만 읽어서
        # "구조는 WavLM-base와 동일하지만 가중치는 랜덤 초기화된" 빈 모델을 만듦
        config = WavLMConfig.from_pretrained(_LOCAL_CONFIG_DIR)    # 로컬 폴더에서 설정값만 로드 (인터넷 불필요)
        self.wavlm = WavLMModel(config)                             # 설정값으로 구조만 생성 (가중치는 랜덤)
        # 아래 가중치(랜덤 초기화 상태)는 inference.py에서 체크포인트(.pt)로
        # 전부 덮어쓰이므로, 이 시점의 값 자체는 결과에 전혀 영향을 주지 않음.
        # 다만 "구조"(층 수, 차원 등)는 체크포인트와 반드시 일치해야 load_state_dict가 성공함.

        # 인코더 층 수를 학습 때와 동일하게 자름 (12층 모델은 전체를 그대로 사용하므로
        # 이 줄은 현재는 아무 것도 잘라내지 않는 항등 연산이지만, 더 적은 층으로
        # 학습한 체크포인트를 쓸 경우를 대비해 명시적으로 남겨둠)
        self.wavlm.encoder.layers = nn.ModuleList(                 # 인코더의 층 리스트를
            list(self.wavlm.encoder.layers)[:num_layers_to_keep]     # 앞에서부터 num_layers_to_keep개만 남겨 재구성
        )

        hidden_size = self.wavlm.config.hidden_size                # WavLM 출력 벡터의 차원 수 조회
        self.attention_pool = AttentionPooling(hidden_size)          # Attention Pooling 층 생성
        self.classifier = nn.Sequential(                             # 분류기: Dropout + 선형층 순서로 구성
            nn.Dropout(0.3),                                           # 학습 때 쓰던 값. 추론 시엔 model.eval()로 자동 비활성화됨
            nn.Linear(hidden_size, num_classes)                        # hidden_size차원 벡터 -> 클래스 수(2)로 매핑
        )

    def forward(self, input_values):
        # input_values: (batch, 오디오 샘플 수) — 16kHz/2.5초 고정 길이로 전처리된 파형
        outputs = self.wavlm(input_values)             # WavLM 인코더에 통과시켜 출력 획득
        hidden_states = outputs.last_hidden_state         # (batch, seq_len, hidden_size) 프레임별 벡터 추출
        pooled = self.attention_pool(hidden_states)         # Attention Pooling으로 발화 전체를 벡터 1개로 요약
        logits = self.classifier(pooled)                     # 분류기를 통과시켜 (batch, num_classes) logit 계산
        return logits                                          # 최종 예측 결과(softmax 적용 전) 반환
