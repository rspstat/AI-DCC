"""
Mission 2 - 신고자/119대원 화자 분류 추론 스크립트

실행 예시:
    python inference.py \
        --audio_dir {wav folder} \
        --label_dir {json folder} \
        --ckpt_path best_model_wavlm_mission2_v9_12layers.pt \
        --output ./outputs/mission2.csv

동작 순서:
    1. label_dir의 모든 json에서 발화 구간(wav 파일명, startAt, endAt) 목록을 만든다.
       (text, speaker 등 다른 필드는 입력으로 읽지 않음 — 텍스트/순서 정보를
        쓰지 않는다는 대회 규정을 지키기 위해 시간 구간 정보만 사용)
       (학습 때는 300ms 미만 발화를 걸러냈지만, 추론은 걸러내지 않는다.
        제출 CSV는 label_dir에 있는 모든 발화에 대해 예측을 제공해야
        채점이 가능하기 때문)
    2. audio_dir에서 해당 wav를 찾아 그 구간만 잘라 16kHz/2.5초 고정 길이로 전처리한다.
       (학습 때 쓴 전처리와 동일 — 서로 다르면 모델이 처음 보는 분포의
        입력을 받게 되어 성능이 크게 떨어짐)
    3. ckpt_path의 학습된 가중치를 불러온다. (model.py는 사전학습 가중치를
       인터넷에서 내려받지 않고 로컬 config로 구조만 만들며, 실제 가중치는
       여기서 체크포인트로 전부 덮어씀 — 채점 환경의 인터넷 연결 여부와 무관하게 동작)
    4. 배치 단위로 추론하여 각 발화의 화자(0=대원, 1=신고자)를 예측한다.
    5. [audio file name, startAt, endAt, speaker] 컬럼의 CSV로 저장한다.
    6. 모델 규모(파라미터 수)와 추론 소요 시간을 화면에 출력한다.
       (계산 효율성 확인용 — 대회 권장 사항)
"""
import argparse                                   # 명령줄 인자(--audio_dir 등)를 처리하기 위해 사용
import glob                                        # 폴더 안 파일 목록을 찾기 위해 사용
import json                                         # json 라벨 파일을 읽기 위해 사용
import os                                            # 경로 조합, 폴더 생성 등에 사용
import time                                           # 추론 소요 시간 측정에 사용
import platform                                        # 실행 환경(OS 등) 확인에 사용

import numpy as np                                  # 배열 연산에 사용
import pandas as pd                                  # 결과를 CSV로 저장하기 위해 사용
import soundfile as sf                               # wav 파일 읽기에 사용
import librosa                                        # 샘플레이트 변환(리샘플링)에 사용
import torch                                           # 모델 실행 및 텐서 연산에 사용
from torch.utils.data import Dataset, DataLoader        # 데이터 파이프라인 구성에 사용
from tqdm import tqdm                                    # 진행률 표시에 사용

from model import WavLMClassifierV2                       # 같은 폴더의 model.py에서 모델 클래스를 가져옴

# 학습 때 쓴 값과 반드시 동일해야 함 — 다르면 모델이 학습 때와 다른 분포의
# 입력을 받아 성능이 무너짐 (학습 코드(.ipynb)의 설정값을 그대로 가져온 것)
SR = 16000                 # WavLM-base가 16kHz로 사전학습되어 있어 고정
MAX_DURATION = 2.5           # 학습 시 모든 발화를 2.5초로 자르거나 패딩했던 것과 동일
NUM_LAYERS_TO_KEEP = 12       # 최종 채택 모델(12층 전체 fine-tuning)의 인코더 층 수

BATCH_SIZE = 16             # Validation 추론 시 사용한 배치 크기 (정확도와는 무관, 속도/메모리에만 영향)


def parse_args():
    parser = argparse.ArgumentParser()                                               # 인자 파서 생성
    parser.add_argument('--audio_dir', required=True, help='wav 파일들이 있는 폴더')      # 오디오 폴더 경로 인자
    parser.add_argument('--label_dir', required=True, help='json(발화 구간 정보) 파일들이 있는 폴더')  # 라벨 폴더 경로 인자
    parser.add_argument('--ckpt_path', required=True, help='학습된 모델 체크포인트(.pt) 경로')  # 체크포인트 경로 인자
    parser.add_argument('--output', required=True, help='결과 CSV 저장 경로')              # 출력 CSV 경로 인자
    return parser.parse_args()                                                          # 파싱된 인자 반환


def build_segment_list(label_dir, audio_dir):
    """label_dir의 모든 json에서 (wav_name, start_ms, end_ms) 목록 생성.
    학습 때와 달리 길이로 걸러내지 않고, label_dir에 있는 모든 발화를 대상으로 한다."""
    json_paths = sorted(glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True))   # label_dir 하위의 모든 json 경로 수집 (정렬해 실행마다 CSV 행 순서가 같도록 함)
    wav_paths = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)      # audio_dir 하위의 모든 wav 경로 수집
    wav_path_map = {os.path.basename(p): p for p in wav_paths}                          # 파일명으로 바로 찾기 위한 매핑 생성

    segments = []          # (wav_name, start_ms, end_ms)를 담을 리스트
    skipped_no_wav = 0       # json은 있는데 대응하는 wav를 못 찾은 경우의 개수
    skipped_bad_time = 0      # startAt/endAt이 없거나 순서가 뒤바뀐 경우의 개수

    for jp in json_paths:                                 # 모든 json 파일을 순회
        try:                                                 # json이 깨져 있어도 전체가 멈추지 않도록 예외 처리
            with open(jp, 'r', encoding='utf-8') as f:          # json 파일 열기
                data = json.load(f)                               # json 내용을 파이썬 객체로 파싱
        except Exception:                                      # 파싱 실패 시
            continue                                              # 이 파일만 건너뛰고 계속 진행

        wav_name = os.path.basename(jp).replace('.json', '.wav')  # json 파일명으로부터 대응 wav 파일명 추정
        if wav_name not in wav_path_map:                           # 대응하는 wav가 없으면
            skipped_no_wav += 1                                      # 카운트만 올리고
            continue                                                  # 이 json은 건너뜀

        for utt in data.get('utterances', []):               # 통화 안의 모든 발화를 순회
            start_ms = utt.get('startAt')                       # 발화 시작 시각
            end_ms = utt.get('endAt')                            # 발화 끝 시각
            if start_ms is None or end_ms is None or end_ms <= start_ms:  # 시간 정보가 비정상이면
                skipped_bad_time += 1                                       # 카운트만 올리고
                continue                                                     # 이 발화는 건너뜀
            segments.append((wav_name, start_ms, end_ms))         # 정상적인 발화만 목록에 추가

    # 몇 개가, 왜 빠졌는지 화면에 남겨서 채점 결과가 예상과 다를 때 원인을 바로 알 수 있게 함
    if skipped_no_wav:                                           # 스킵된 게 있으면
        print(f"[inference.py] 경고: 대응하는 wav를 찾지 못해 스킵된 json {skipped_no_wav}개")  # 경고 출력
    if skipped_bad_time:                                          # 시간 오류로 스킵된 게 있으면
        print(f"[inference.py] 경고: startAt/endAt이 유효하지 않아 스킵된 발화 {skipped_bad_time}개")  # 경고 출력

    return segments, wav_path_map                                 # 발화 목록과 wav 경로 매핑 반환


class InferenceDataset(Dataset):
    """발화 구간 목록을 받아, 모델 입력 형태(16kHz/2.5초 고정 길이 파형)로 변환."""

    def __init__(self, segments, wav_path_map, sr=SR, max_duration=MAX_DURATION):  # 초기화 함수
        self.segments = segments                              # 발화 구간 목록 저장
        self.wav_path_map = wav_path_map                        # 파일명->경로 매핑 저장
        self.sr = sr                                             # 목표 샘플레이트 저장
        self.max_len = int(sr * max_duration)                     # 목표 길이를 샘플 개수로 환산해 저장
        self.load_fail_count = 0                                   # 오디오 로딩 실패 횟수 (추론 후 main()에서 보고)

    def __len__(self):                           # 전체 데이터 개수를 반환하는 필수 메서드
        return len(self.segments)                  # 발화 개수 반환

    def __getitem__(self, idx):                                     # 인덱스 하나에 대한 데이터를 반환하는 필수 메서드
        wav_name, start_ms, end_ms = self.segments[idx]                # idx번째 발화 정보 꺼내기
        wav_path = self.wav_path_map[wav_name]                          # 해당 wav의 전체 경로 조회

        try:                                                              # 오디오 로딩 중 에러 발생 가능성에 대비해 예외 처리
            # wav 파일 전체를 읽지 않고, 필요한 구간(start_ms~end_ms)만 읽어서 속도를 아낌
            info = sf.info(wav_path)                                        # 파일 메타정보(샘플레이트 등) 조회
            native_sr = info.samplerate                                      # 원본 파일의 실제 샘플레이트
            y, orig_sr = sf.read(                                             # 필요한 구간만 읽기
                wav_path,
                start=int(start_ms / 1000 * native_sr),                         # 시작 시각을 프레임 번호로 환산
                stop=int(end_ms / 1000 * native_sr),                            # 끝 시각도 프레임 번호로 환산
                dtype='float32'                                                  # float32 형식으로 읽기
            )
            # WavLM은 16kHz 입력을 전제로 사전학습되어 있으므로, 원본이 다른
            # 샘플레이트면 반드시 16kHz로 맞춰야 함 (음높이/속도는 그대로 유지되는 변환)
            if orig_sr != self.sr:                                            # 원본 샘플레이트가 목표와 다르면
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)      # 16kHz로 리샘플링
        except Exception:                                                   # 파일 손상 등으로 못 읽으면
            # 전체 파이프라인이 멈추지 않도록 무음(0)으로 채운 벡터로 대체하고 카운트만 남김
            y = np.zeros(self.max_len, dtype=np.float32)                       # 무음 벡터로 대체
            self.load_fail_count += 1                                           # 실패 횟수 1 증가

        # sf.read가 예외 없이 빈 배열을 반환하는 경우(구간이 파일 끝을 벗어난 경우 등)에 대한 안전장치
        if y is None or len(y) == 0:                                        # 읽은 결과가 비어있으면
            y = np.zeros(self.max_len, dtype=np.float32)                       # 무음 벡터로 대체

        # 학습 때와 동일하게 모든 발화를 2.5초 고정 길이로 맞춤 (짧으면 무음 패딩, 길면 앞부분만 사용)
        if len(y) < self.max_len:                                           # 목표 길이보다 짧으면
            y = np.pad(y, (0, self.max_len - len(y)))                          # 뒤쪽을 0으로 채워 길이를 맞춤
        else:                                                                 # 목표 길이보다 길거나 같으면
            y = y[:self.max_len]                                               # 앞부분만 잘라서 사용

        # 학습 때와 동일한 정규화 (평균 0, 표준편차 1) — 녹음 환경별 음량 차이를 줄여줌
        y = (y - y.mean()) / (y.std() + 1e-7)                                # 평균/표준편차로 정규화 (1e-7은 0 나누기 방지)
        return torch.tensor(y, dtype=torch.float32)                          # 파형을 텐서로 변환해 반환


def count_parameters(model):
    """모델의 Total 파라미터 수와 Active(학습 시 fine-tuning 대상이었던) 파라미터 수를 센다.
    대회 권장 사항인 'Total/Active 파라미터 수' 보고를 위해 사용.

    Active 기준: 학습 코드(.ipynb)에서 requires_grad=True였던 부분.
        학습 때 고정(freeze)했던 것은 아래 세 가지뿐이므로, 전체에서 이들을 뺀 값이 Active임
        - CNN 특징 추출기 (wavlm.feature_extractor)
        - 인코더 positional conv (wavlm.encoder.pos_conv_embed)
        - 인코더 마지막 layer_norm (wavlm.encoder.layer_norm)
    ※ 추론 시에는 Total 전체가 연산에 참여한다."""
    total = sum(p.numel() for p in model.parameters())                         # 전체 파라미터 개수 합산
    frozen = sum(p.numel() for m in (model.wavlm.feature_extractor,             # 학습 때 고정했던 모듈들의 파라미터 합
                                     model.wavlm.encoder.pos_conv_embed,
                                     model.wavlm.encoder.layer_norm)
                 for p in m.parameters())
    return total, total - frozen                                               # (총 파라미터 수, 학습 대상 파라미터 수) 반환


def main():
    args = parse_args()                                                  # 명령줄 인자 파싱
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 있으면 GPU, 없으면 CPU 사용

    # ---- 컴퓨팅 환경 정보 출력 (대회 권장 사항: 학습/추론 환경 명시) ----
    print(f"[inference.py] device: {device}")                              # 사용 중인 디바이스 출력
    if torch.cuda.is_available():                                          # GPU를 쓰는 경우
        print(f"[inference.py] GPU: {torch.cuda.get_device_name(0)}")        # GPU 모델명 출력
    else:                                                                    # CPU만 쓰는 경우
        print(f"[inference.py] CPU: {platform.processor() or platform.machine()}")  # CPU 정보 출력
    print(f"[inference.py] Validation 추론 batch size: {BATCH_SIZE}")        # 배치 크기 출력

    # 1. label_dir에서 예측 대상 발화 목록 만들기
    segments, wav_path_map = build_segment_list(args.label_dir, args.audio_dir)  # 발화 목록, wav 경로 매핑 생성
    print(f"[inference.py] 예측 대상 발화 수: {len(segments)}")                  # 총 발화 개수 출력

    if len(segments) == 0:                                                 # 대상 발화가 하나도 없으면
        raise RuntimeError("예측 대상 발화가 0개입니다. audio_dir/label_dir 경로를 확인하세요.")  # 명확한 에러로 중단

    # 2. 전처리 파이프라인 구성
    dataset = InferenceDataset(segments, wav_path_map)                      # Dataset 인스턴스 생성
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 순서 유지, 단일 프로세스로 로딩

    # 3. 모델 생성 후 학습된 가중치 로드
    model = WavLMClassifierV2(num_classes=2, num_layers_to_keep=NUM_LAYERS_TO_KEEP).to(device)  # 모델 생성 후 device로 이동
    state_dict = torch.load(args.ckpt_path, map_location=device, weights_only=True)  # 체크포인트 로드 (가중치만 안전하게)
    model.load_state_dict(state_dict)                                       # 모델에 체크포인트 가중치 적용
    model.eval()                                                             # Dropout 등을 추론 모드로 전환

    # ---- 파라미터 수 출력 (대회 권장 사항) ----
    total_params, active_params = count_parameters(model)                   # 총 / 학습 대상 파라미터 수 계산
    print(f"[inference.py] Total 파라미터 수: {total_params:,}")               # 총 파라미터 수 출력
    print(f"[inference.py] Active 파라미터 수: {active_params:,} "
          f"(전체의 {active_params / total_params * 100:.2f}%, 학습 때 fine-tuning한 부분 — 추론 시에는 전체 파라미터가 연산에 관여함)")

    # 4. 배치 단위 추론 (+ 추론 시간 측정)
    all_preds = []                                                          # 예측 결과를 모을 리스트
    inference_start = time.time()                                           # 추론 시작 시각 기록
    with torch.no_grad():                                                   # 추론만 하므로 그래디언트 계산을 꺼서 메모리/속도를 아낌
        for batch in tqdm(loader, desc="추론 중"):                            # 배치를 하나씩 꺼내며 진행률 표시
            batch = batch.to(device)                                          # 입력을 device로 이동
            outputs = model(batch)                                            # 모델 예측(logit) 획득
            preds = outputs.argmax(dim=1).cpu().numpy()                       # 확률이 더 높은 쪽(대원/신고자) 선택
            all_preds.extend(preds.tolist())                                   # 예측 결과를 리스트에 누적
    inference_elapsed = time.time() - inference_start                        # 전체 추론에 걸린 시간 계산

    # ---- 추론 시간 출력 (대회 권장 사항: 전체/샘플당 평균 추론시간) ----
    per_sample_ms = inference_elapsed / len(segments) * 1000                 # 샘플 1개당 평균 소요시간(ms) 계산
    print(f"[inference.py] Validation 전체 추론시간: {inference_elapsed:.2f}초 "
          f"({len(segments)}개 발화, batch_size={BATCH_SIZE}, device={device})")
    print(f"[inference.py] 샘플당 평균 추론시간: {per_sample_ms:.2f} ms")

    if dataset.load_fail_count:                                             # 오디오 로딩 실패가 있었으면
        print(f"[inference.py] 경고: 오디오 로딩 실패로 무음 처리된 발화 {dataset.load_fail_count}개")  # 경고 출력

    # 5. 결과를 지정된 CSV 형식으로 저장
    rows = [                                                                 # CSV의 각 행을 딕셔너리로 구성
        {
            'audio file name': wav_name,                                      # 오디오 파일명
            'startAt': start_ms,                                               # 발화 시작 시각
            'endAt': end_ms,                                                    # 발화 끝 시각
            'speaker': int(pred)                                                 # 예측된 화자(0=대원, 1=신고자)
        }
        for (wav_name, start_ms, end_ms), pred in zip(segments, all_preds)       # 발화 정보와 예측 결과를 짝지어 순회
    ]
    df = pd.DataFrame(rows, columns=['audio file name', 'startAt', 'endAt', 'speaker'])  # 지정된 컬럼 순서로 DataFrame 생성

    output_dir = os.path.dirname(args.output)                                # 출력 경로의 상위 폴더 추출
    if output_dir:                                                            # 상위 폴더 경로가 있으면(최상위가 아니면)
        os.makedirs(output_dir, exist_ok=True)                                  # 폴더가 없으면 생성
    df.to_csv(args.output, index=False, encoding='utf-8')                     # 인덱스 없이 CSV 저장 (내용이 영문/숫자뿐이라 BOM 없는 utf-8 사용)

    print(f"[inference.py] 결과 저장 완료: {args.output} ({len(df)}개 행)")       # 저장 완료 메시지 출력


if __name__ == '__main__':                 # 이 스크립트가 직접 실행될 때만 main()을 호출
    main()                                     # 메인 함수 실행
