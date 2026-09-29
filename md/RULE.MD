## Rule

1. 대학부 미션 수행을 위해 제공된 데이터셋 



Training 데이터셋: 29,200개
Validation 데이터셋: 3,640개


2. 제출파일 및 채점 기준 안내

1) 제출물 구성
- 예선 출제문제 12페이지 기준으로 제출하되, pptx는 미션 1,2,3을 모두 포함한 단일 파일로 생성할 것(분량 자유)
- 미션별 코드 폴더를 만들어 총 3개의 폴더로 제출. 즉, 미션별 코드 폴더 3개와 pptx 1개를 최종 제출할 것
- 미션별 inference.py 실행 한 번으로 해당 미션의 결과 파일을 생성할 수 있어야 함
- 정해진 파일명/폴더구조는 따로 없음

2) 채점 환경
- Pre-trained 모델을 사용하는 경우 채점 시점에 인터넷 다운로드에 의존하면 실행 실패 가능성이 있으므로, 
필요한 tokenizer, config, weight 파일을 제출물에 포함하거나 로컬 경로에서 불러오도록 구성할 것
- requirements.txt로 실행 환경을 재현할 수 있어야 함
- README.md는 필수 제출물은 아니나, 실행 방법은 README 또는 코드 주석으로 명확히 설명하는 것을 권장함
- 학습 과정과 주요 실험 결과는 ipynb 또는 PPT에 정리할 것
- 추론 시간 관련 안내
* 모델의 계산 효율성을 확인할 수 있도록 다음 정보를 함께 명시할 것을 권장함.
(1) 모델의 Total 파라미터 수 및 Active 파라미터 수
(2) 학습 및 추론에 사용한 컴퓨팅 환경 (GPU/CPU 종류 등)
(3) Validation 추론 시 batch size
(4) 해당 환경에서 측정한 Validation 데이터셋 전체 추론시간 또는 샘플당 평균 추론시간

3. Decision Threshold 관련 기준

Decision Threshold 관련하여 여러 팀에서 문의가 있어 기준을 아래와 같이 정리하여 안내드립니다.


Q1. Decision Threshold는 반드시 0.5로 사용해야 하나요?

네. 최종 예측 확률을 label 0 또는 1로 변환할 때 사용하는 Decision Threshold는 Train / Validation 여부와 관계없이 0.5로 고정하여 사용해주시기 바랍니다.

즉,

* prediction ≥ 0.5 → 1
* prediction < 0.5 → 0

의 기준을 동일하게 적용해주시면 됩니다.



Q2. Validation set을 사용하지 않고 Train set에서 별도의 subset을 나누어 Threshold를 찾는 것은 가능한가요?

허용되지 않습니다.

Train set을 9:1 등으로 나누거나 OOF prediction을 활용하더라도, 해당 결과를 바탕으로 
최종 Decision Threshold를 0.4, 0.6 등 0.5가 아닌 값으로 조정하는 것은 동일하게 Threshold Tuning에 해당합니다.

이번 미션에서는 Threshold를 결정하는 데이터가 Train인지 Validation인지와 관계없이 최종 Decision Threshold 자체를 0.5로 고정하는 것이 기준입니다.



Q3. OOF prediction을 사용하는 것 자체가 금지되나요?

아닙니다.

OOF를 모델 학습, 앙상블, 성능 분석, 확률 보정 등의 목적으로 활용하는 것은 가능합니다.

다만 OOF prediction을 이용하여 “이 모델은 0.37에서 F1이 가장 높으므로 Threshold를 0.37로 사용하자”와 같이 
최종 Decision Threshold를 결정하는 용도로 사용하는 것은 허용되지 않습니다.



Q4. Calibration을 적용하는 것은 가능한가요?

가능합니다.

모델이 출력하는 확률값 자체가 보다 적절한 확률값이 되도록 학습하거나 보정하는 것은 허용됩니다.

예를 들어,

* Training set 내부의 별도 subset 또는 OOF prediction을 활용한 probability calibration
* Calibration을 고려한 loss 설계
* ECE 등 calibration 관련 정규화 항 추가

등은 사용할 수 있습니다.

다만 calibration 이후에도 최종 label을 결정하는 Threshold는 0.5로 고정되어야 합니다.

즉, 모델이 산출하는 확률값의 품질을 개선하는 것은 가능하지만, 그 확률값을 0과 1로 나누는 기준선 자체를 이동시키는 것은 허용되지 않습니다.


Q5. Class마다 서로 다른 Threshold를 사용하는 것은 가능한가요?

허용되지 않습니다.

Multi-label classification에서 각 class별 최적 Threshold를 별도로 탐색하여 적용하는 방식 역시 Threshold Tuning에 해당합니다.

모든 class에 동일하게 0.5의 Decision Threshold를 적용해주시기 바랍니다.



최종 정리

이번 미션의 기준은 다음과 같습니다.

“확률값을 더 잘 예측하기 위한 모델 학습 및 calibration은 자유롭게 수행할 수 있으나, 최종 확률을 label로 변환하는 Decision Threshold는 항상 0.5로 고정한다.”