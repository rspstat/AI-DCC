"""
Mission 1 신고자 성별 분류 모델
- ECAPA-TDNN(SpeechBrain, VoxCeleb 사전학습, 가중치 고정)으로 신고자 발화별 임베딩(192차원) 추출
- 발화 길이 가중 평균으로 통화 단위 벡터 생성
- 로지스틱 회귀(StandardScaler 포함)로 성별 분류
"""
import os, json, tempfile
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import joblib
from sklearn.preprocessing import normalize
from speechbrain.inference.speaker import EncoderClassifier

SR_IN, SR_MODEL = 8000, 16000   # 원본 8kHz → ECAPA 입력 16kHz
MIN_SEC = 0.3                   # 0.3초 미만 발화 제외 (학습과 동일)
LABELS = {1: "F", 0: "M"}       # 공통 규칙: F=1, M=0


class GenderClassifier:
    def __init__(self, clf_path, device):
        self.device = device
        ecapa_dir = os.path.join(os.path.dirname(os.path.abspath(clf_path)), "ecapa")
        # 로컬 weight만 사용 (인터넷 다운로드 없음)
        self.encoder = EncoderClassifier.from_hparams(
            source=ecapa_dir,
            savedir=os.path.join(tempfile.gettempdir(), "ecapa_run"),
            overrides={"pretrained_path": ecapa_dir},
            run_opts={"device": "cuda:0" if device.type == "cuda" else "cpu"},
        )
        self.encoder.eval()
        self.clf = joblib.load(clf_path)   # StandardScaler + LogisticRegression

    def load_segments(self, wav_path, json_path):
        """신고자(speaker=1) 발화 구간을 16kHz로 잘라 반환. 입력으로는 startAt, endAt, speaker만 사용"""
        audio, _ = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio16 = AF.resample(torch.tensor(audio).to(self.device), SR_IN, SR_MODEL)

        segs = []
        if json_path and os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                lab = json.load(f)
            for u in lab.get("utterances", []):
                if u.get("speaker") != 1:
                    continue
                s = audio16[int(u["startAt"] * 16): int(u["endAt"] * 16)]   # 16kHz: 1ms = 16샘플
                if len(s) > 0:
                    segs.append(s)
        if not segs:
            segs = [audio16]    # 라벨이 없거나 신고자 발화가 없으면 통화 전체 사용

        kept = [s for s in segs if len(s) >= MIN_SEC * SR_MODEL]
        if not kept:
            kept = [max(segs, key=len)]   # 전부 짧으면 가장 긴 발화 1개 유지
        return kept

    @torch.no_grad()
    def embed(self, segs):
        """발화들을 padding해 한 번에 임베딩"""
        lens = torch.tensor([len(s) for s in segs], dtype=torch.float32)
        batch = torch.nn.utils.rnn.pad_sequence(segs, batch_first=True)
        emb = self.encoder.encode_batch(batch, (lens / lens.max()).to(self.device))
        return emb.squeeze(1).cpu().numpy(), lens.numpy() / SR_MODEL

    def predict(self, wav_path, json_path):
        """통화 하나의 성별 예측. (라벨 문자열, 여성 확률) 반환"""
        segs = self.load_segments(wav_path, json_path)
        emb, dur = self.embed(segs)
        x = np.average(normalize(emb), axis=0, weights=dur)   # 발화 길이 가중 평균
        prob_F = float(self.clf.predict_proba(x[None, :])[0, 1])
        return LABELS[int(prob_F >= 0.5)], prob_F
