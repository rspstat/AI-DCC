"""
[1회성 준비 스크립트 — 채점 시에는 실행되지 않습니다]

model.py는 채점 환경이 인터넷에 연결되지 않아도 동작하도록, Hugging Face Hub에서
가중치를 받지 않고 '구조(config)'만 로컬 폴더에서 읽습니다.
이 스크립트는 그 로컬 config 폴더(pretrained_config/wavlm-base/config.json)를
인터넷이 되는 환경에서 한 번 만들어 둡니다.

실행:
    python export_wavlm_config.py

결과:
    ./pretrained_config/wavlm-base/config.json   (model.py와 같은 위치에 두어야 함)

가중치(weight)는 저장하지 않습니다. 실제 가중치는 inference.py가 학습된 체크포인트(.pt)로
전부 덮어쓰기 때문에 사전학습 원본 가중치는 추론에 필요하지 않습니다.
"""
import os                                                                    # 저장 경로를 조합하기 위해 사용
from transformers import WavLMConfig                                          # 가중치 없이 구조 설정만 담는 클래스

HERE = os.path.dirname(os.path.abspath(__file__))                              # 이 스크립트가 있는 폴더
SAVE_DIR = os.path.join(HERE, "pretrained_config", "wavlm-base")                # model.py가 기대하는 위치

config = WavLMConfig.from_pretrained("microsoft/wavlm-base")                    # Hub에서 설정값(config.json)만 내려받음
config.save_pretrained(SAVE_DIR)                                                 # 로컬 폴더에 저장
print(f"저장 완료: {SAVE_DIR}")                                                   # 결과 경로 안내
