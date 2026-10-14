"""Mission 3 공통 정의: 라벨, 입력 텍스트 구성, KLUE-RoBERTa 모델 구조"""
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# 공식 9개 증상, 순서 고정 (팀 통일안)
TARGET_SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움",
                   "열상", "오심", "전신쇠약", "호흡곤란"]
THRESHOLD = 0.5      # 대회 규정: 모든 클래스 0.5 고정, prob >= 0.5 -> 1
MAX_LEN = 512


def build_input_text(data):
    """utterances를 startAt 순으로 정렬해 text만 공백으로 연결 (화자 태그 없음, 규정)"""
    utts = sorted(data['utterances'], key=lambda u: u['startAt'])
    return " ".join(u['text'] for u in utts)


class KlueSymptomClassifier(nn.Module):
    """KLUE-RoBERTa + mean pooling + MLP head (학습 코드와 동일 구조)

    config_dir: 로컬에 저장한 config 폴더 → 인터넷 없이 구조만 만들고 가중치는 state_dict로 로드
    """
    def __init__(self, config_dir, num_classes=9, dropout=0.3):
        super().__init__()
        cfg = AutoConfig.from_pretrained(config_dir)
        self.backbone = AutoModel.from_config(cfg, add_pooling_layer=False)
        h = cfg.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        return self.classifier((x * m).sum(1) / m.sum(1).clamp(min=1e-9))
