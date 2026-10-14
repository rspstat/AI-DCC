"""
Mission 1 추론 스크립트
실행: python inference.py --audio_dir {wav 폴더} --label_dir {json 폴더}
                          --ckpt_path {weights/clf.joblib} --output ./outputs/mission1.csv
- ckpt_path와 같은 폴더의 ecapa/ 에서 ECAPA 가중치를 불러옴
- 출력 CSV: [audio file name], [gender]
"""
import os, sys, csv, glob, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 채점 환경: 인터넷 다운로드 방지
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import GenderClassifier

WRITE_HEADER = True   # CSV 헤더 포함 여부


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission1.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenderClassifier(args.ckpt_path, device)

    # 파일명(확장자 제외)으로 wav와 json 매칭
    wavs = sorted(glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True))
    jsons = {os.path.splitext(os.path.basename(j))[0]: j
             for j in glob.glob(os.path.join(args.label_dir, "**", "*.json"), recursive=True)}
    print(f"device={device}, wav {len(wavs)}개, json {len(jsons)}개")

    rows, t0 = [], time.time()
    for i, w in enumerate(wavs, 1):
        stem = os.path.splitext(os.path.basename(w))[0]
        gender, _ = model.predict(w, jsons.get(stem))
        rows.append([os.path.basename(w), gender])
        if i % 500 == 0:
            print(f"{i}/{len(wavs)} 처리")
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if WRITE_HEADER:
            writer.writerow(["audio file name", "gender"])
        writer.writerows(rows)

    print(f"저장: {args.output}")
    print(f"전체 추론 시간 {elapsed:.1f}초, 통화당 평균 {elapsed / max(len(wavs), 1) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
