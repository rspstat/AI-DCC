"""
Mission 3 (환자 증상 인식) 추론 - KLUE-RoBERTa(posw) + KcELECTRA-v2022(none) 앙상블

실행 (대회 지정 형식)
  python inference.py --audio_dir {wav} --label_dir {json} --ckpt_path ./ckpt --output ./outputs/mission3.csv

앙상블 규칙 (Validation으로 가중치를 고르지 않음)
  1) 모델(백본)별로 seed 42/123/2026 확률 평균
  2) 두 모델 확률을 1:1 평균
  3) 모든 클래스 threshold 0.5 고정 (prob >= 0.5 → 해당 증상)

입력: 라벨 JSON의 utterances를 startAt 순으로 정렬해 text만 연결 (화자 태그·다른 필드 미사용)
--audio_dir는 대회 실행 형식을 맞추기 위한 인자이며 Mission 3에서는 사용하지 않음
인터넷 없이 --ckpt_path 안의 토크나이저·config·가중치만으로 동작
"""
import argparse
import glob
import json
import os
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')          # 오프라인 채점 환경
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from model_defs import MAX_LEN, TARGET_SYMPTOMS, THRESHOLD, KlueSymptomClassifier, build_input_text

SEEDS = [42, 123, 2026]


def load_texts(label_dir, with_labels=False):
    """라벨 JSON을 파일명 순으로 읽어 입력 텍스트 생성

    with_labels=True(검증용 --save_probs)일 때만 정답 symptom을 읽음. 예측에는 사용하지 않음
    """
    files = sorted(glob.glob(os.path.join(label_dir, '**', '*.json'), recursive=True))
    names, texts, labels = [], [], []
    for fp in files:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        names.append(os.path.basename(fp))
        texts.append(build_input_text(data))
        if with_labels:
            labels.append([1 if s in data.get('symptom', []) else 0 for s in TARGET_SYMPTOMS])
    return names, texts, np.array(labels)


def make_loader(texts, tokenizer, batch_size, use_token_type):
    """max_length 512로 자르고, 배치 안에서 가장 긴 길이에 맞춰 padding"""
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, return_token_type_ids=use_token_type)
    items = [{k: enc[k][i] for k in enc.keys()} for i in range(len(texts))]
    collate = lambda b: tokenizer.pad(b, padding='longest', return_tensors='pt')
    return DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device, hf_model):
    """sigmoid 확률 (샘플 수, 9) 반환. GPU에서는 fp16 autocast"""
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(**batch).logits if hf_model else model(batch['input_ids'], batch['attention_mask'])
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def run_family(name, texts, ckpt_dir, device, batch_size):
    """한 백본의 seed 3개 확률 평균"""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    hf_model = (name == 'kcelectra')
    loader = make_loader(texts, tokenizer, batch_size, use_token_type=False)   # 두 모델 모두 학습 때 미사용
    probs = []
    for seed in SEEDS:
        if hf_model:
            # KcELECTRA: HF 기본 분류 head (로컬 config로 구조 생성)
            model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(ckpt_dir))
        else:
            # KLUE-RoBERTa: mean pooling + MLP head (model_defs.py)
            model = KlueSymptomClassifier(ckpt_dir)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'seed{seed}.pt'), map_location='cpu'), strict=True)
        model.to(device)
        t0 = time.time()
        probs.append(predict(model, loader, device, hf_model))
        print(f"  {name} seed{seed}: {time.time() - t0:.1f}s", flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio_dir', default=None, help='Mission 3에서는 사용하지 않음')
    ap.add_argument('--label_dir', required=True)
    ap.add_argument('--ckpt_path', required=True, help='klue/, kcelectra/ 하위 폴더를 가진 체크포인트 폴더')
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--save_probs', default=None, help='(검증용) 앙상블 확률을 팀 CSV 형식으로 저장할 경로')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_start = time.time()

    names, texts, labels = load_texts(args.label_dir, with_labels=bool(args.save_probs))
    print(f"라벨 파일 {len(names)}개, device={device}")

    p_klue = run_family('klue', texts, os.path.join(args.ckpt_path, 'klue'), device, args.batch_size)
    p_kc = run_family('kcelectra', texts, os.path.join(args.ckpt_path, 'kcelectra'), device, args.batch_size)
    prob = (p_klue + p_kc) / 2                                   # 두 백본 1:1
    pred = prob >= THRESHOLD                                     # 모든 클래스 0.5 고정

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 라벨 JSON의 symptom과 같은 리스트 문자열 (큰따옴표), 증상이 없으면 []
    symptoms = [json.dumps([s for s, on in zip(TARGET_SYMPTOMS, row) if on], ensure_ascii=False) for row in pred]
    pd.DataFrame({'label file name': names, 'symptom': symptoms}).to_csv(
        args.output, index=False, encoding='utf-8-sig')

    if args.save_probs:
        df = pd.DataFrame({'call_id': [os.path.splitext(n)[0] for n in names]})
        for i, s in enumerate(TARGET_SYMPTOMS):
            df[f'prob_{s}'] = prob[:, i]
        for i, s in enumerate(TARGET_SYMPTOMS):
            df[f'true_{s}'] = labels[:, i]
        df.to_csv(args.save_probs, index=False, encoding='utf-8-sig')

    elapsed = time.time() - t_start
    print(f"완료: {args.output} ({len(names)}행) | 전체 {elapsed:.1f}s, 샘플당 {elapsed / max(len(names), 1) * 1000:.0f}ms")


if __name__ == '__main__':
    main()
