# MiniMind-V 색상과 도형 SFT

## 목적

공개 VLM을 작은 합성 데이터로 추가 학습하고, 이미지에 근거한 색상·도형 답변을
학습 전후 동일한 테스트셋으로 비교한다.

## 모델과 입력

- 모델: `caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`
- 고정 리비전: `4e6f87f184b27f0692e1f8b3afed959de254894b`
- 공개 번들의 `inference.py`, `minimind_v/`, 토크나이저, 설정, 언어/Projector 가중치와 별도 비전 인코더를 사용한다.
- 256×256 흰 배경, 빨강·초록·파랑 중 하나, 원·사각형 중 하나.
- 위치와 크기를 seed 42로 바꾸고, 이미지 픽셀 SHA256 중복은 모든 분할에서 제외한다.
- 학습 600장, 검증 120장, 테스트 120장. 각 분할에서 6개 조합의 수를 같게 한다.
- 질문: `What color and shape is it? Answer with the color and shape only.`
- 답변: `red circle`, `red square`, `green circle`, `green square`, `blue circle`, `blue square`.

## 절차

1. 공개 리비전의 파일을 실행 폴더의 `base_model/`에 준비한다. 원본 캐시 가중치를
   읽기 전용으로 참조할 수 있으나, 다른 컴퓨터로 옮길 때는 참조 대상 파일도 보관한다.
2. `projects/minimind-v/`에서 실행한다. CUDA를 사용할 수 있는 PyTorch,
   Transformers 4.57.6, Pillow, safetensors 환경이 필요하다.

   ```bash
   python scripts/train_shapes.py \
     --run-dir /absolute/repository/outputs/minimind-v/<실행-ID> \
     --epochs 8 --batch-size 16
   ```

3. Projector만 학습한다. 학습률 3e-4, AdamW, BF16 autocast, 답변/EOS만 CE 대상으로 삼는다.
4. 검증 CE가 가장 낮은 에포크를 선택한다. 테스트 결과로 체크포인트를 선택하지 않는다.
5. 비전 인코더와 언어 모델의 학습 전후 매개변수 SHA256이 같은지 확인한다.
6. 저장한 native 체크포인트를 재로딩하고 테스트셋을 평가한다.

## 판정

- 생성: greedy, 최대 12토큰. 앞뒤 공백과 끝의 마침표·느낌표를 정규화한 완전 일치율을 기록한다.
- 색상 정확도와 도형 정확도를 별도로 기록한다.
- 조합 정확도 90%를 목표로 삼되, 실제 달성 여부는 실행 결과로 판정한다.
- 검은 이미지에서는 원래 정답 정확도가 떨어지는지 확인한다.
- 이미지 교체에서는 원래 정답 정확도와 교체된 이미지 정답 정확도를 함께 확인한다.
  교체 대상은 다른 색상·도형 조합이며, 동일한 고정 질문을 사용한다.
- 고정 매개변수 불변, 유한한 loss/gradient, 체크포인트 재로딩을 필수 실행 검증으로 삼는다.

## 산출물과 한계

이미지와 정답, 에포크별 CE/검증 예측, 학습 전후 원본·검은 이미지·교체 이미지의
전체 응답, 가중치, 환경 정보와 완료 기록을 `outputs/minimind-v/<실행-ID>/`에 둔다.
보고서는 `reports/minimind-v/<실행-ID>.md`에 쓴다.

이 실험의 테스트는 같은 생성 규칙에서 뽑은 새로운 위치·크기의 이미지다.
자연 사진, 다른 배경색, 보지 못한 색상·도형 조합의 일반화 성능을 뜻하지 않는다.
CPU 전용 환경에서는 이 GPU 실행 스크립트가 중단된다.
