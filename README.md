# LLM Playground

모델 구현과 학습용 노트북을 함께 관리하는 저장소입니다.

| 경로 | 용도 |
| --- | --- |
| [notebooks/gpt/](notebooks/gpt/README.md) | GPT 챕터별 실습과 Attention·모델 구조 실험 |
| [notebooks/minimind/](notebooks/minimind/README.md) | MiniMind 구조·학습·멀티모달 marimo 실습 |
| [notebooks/physicalai/](notebooks/physicalai/README.md) | MuJoCo 시뮬레이션 로봇팔로 따라가는 Physical AI 9장 marimo 실습 |
| [notebooks/unsloth/](notebooks/unsloth/README.md) | Unsloth 기반 Qwen 학습·배포 실습 |
| [notebooks/vlm-from-scratch/](notebooks/vlm-from-scratch/README.md) | VLM 구성·학습·멀티모달 응용 12부 marimo 실습 |
| `projects/minimind*/` | MiniMind 계열의 독립 모델 구현·학습·추론 프로젝트 |
| `references/slides/` | 교재와 슬라이드 |
| `test_scenarios/<프로젝트>/` | 재사용할 테스트 입력·절차·판정 기준 |
| `reports/<프로젝트>/` | 검증·분석 보고서 |
| `outputs/<프로젝트>/<실행 ID>/` | 실행 산출물과 로그, Git 제외 |

각 실행 디렉토리의 README를 먼저 확인하세요. `pyproject.toml`, `uv.lock`,
가상 환경은 프로젝트별로 관리하며, 해당 디렉토리에서 `uv sync --locked`를 실행합니다.

```bash
# 저장소 루트에서 시작
cd notebooks/gpt
uv sync --locked
uv run marimo edit .
```

## 교재

교재 PDF는 `references/slides/` 폴더에 있습니다.

- [slides.pdf](references/slides/slides.pdf)
- [slides-part2.pdf](references/slides/slides-part2.pdf)
  
## 챕터별 실습 노트북

- [Chapter_2_Exercise_Dataset.py](notebooks/gpt/Chapter_2_Exercise_Dataset.py)
  - Dataset, DataLoader 만드는법
- [Chapter_3_Excercise_Attention.py](notebooks/gpt/Chapter_3_Excercise_Attention.py)
  - CausalAttention의 구조
- [Chapter_4_Excercise_GPT.py](notebooks/gpt/Chapter_4_Excercise_GPT.py)
  - TransformerBlock, GPT의 구조
- [Chapter_5_Excercise_Pretraining.py](notebooks/gpt/Chapter_5_Excercise_Pretraining.py)
  - Cross Entropy, 모델 학습과정
- [Chapter_6_Excercise_Finetuning_Classification.py](notebooks/gpt/Chapter_6_Excercise_Finetuning_Classification.py)
- [Chapter_6_Excercise_Finetuning_Classification_LoRA.py](notebooks/gpt/Chapter_6_Excercise_Finetuning_Classification_LoRA.py)
  - LoRALayer만드는 방법과 Linear Layer에 LoRA를 적용하는 방법
- [Chapter_7_Exercise_Follow_Instructions.py](notebooks/gpt/Chapter_7_Exercise_Follow_Instructions.py)
- [Chapter_7_Exercise_Follow_Instructions_dpo.py](notebooks/gpt/Chapter_7_Exercise_Follow_Instructions_dpo.py)

## minimind

작은 모델과 PyTorch 구현으로 언어 모델부터 이미지·음성 멀티모달 모델까지 학습하는 교육·실습용 프로젝트입니다.

- **[MiniMind](https://github.com/jingyaogong/minimind)** — 텍스트를 생성하는 언어 모델(LLM). 사전 학습, SFT, LoRA, 지식 증류, 선호 최적화·강화 학습과 도구 호출을 실습합니다. [한국어 안내](projects/minimind/README_ko.md)
- **[MiniMind-V](https://github.com/jingyaogong/minimind-v)** — MiniMind에 비전 인코더와 Projector를 연결한 시각·언어 모델(VLM). 이미지와 질문을 입력받아 텍스트로 답하며, 이미지 특징 결합과 시각 지시 학습을 실습합니다. [한국어 안내](projects/minimind-v/README_ko.md)
- **[MiniMind-O](https://github.com/jingyaogong/minimind-o)** — 텍스트·이미지·음성을 이해하고 텍스트와 스트리밍 음성을 출력하는 Omni 모델. Thinker–Talker 구조와 음성 생성, 실시간 음성 상호작용을 실습합니다. [한국어 안내](projects/minimind-o/README_ko.md)

**권장 학습 순서:** MiniMind → MiniMind-V → MiniMind-O. 환경 준비와 실행 방법은 각 프로젝트의 한국어 안내를 참고합니다.

## unsloth

Unsloth를 활용해 Qwen 계열 모델의 미세 조정, 강화 학습, 증류와 배포를 실습합니다. 참고 저장소: [Unsloth 공식 노트북 모음](https://github.com/unslothai/notebooks).

- **[Qwen3_5_(4B)-Decision.py](notebooks/unsloth/Qwen3_5_%284B%29-Decision.py)** — 입력에 대한 여러 질문의 선택형·참거짓·등급 판단을 한 번의 순전파로 처리하고, 판단 헤드와 LoRA를 학습합니다.
- **[Qwen3_5_(4B)_Vision.py](notebooks/unsloth/Qwen3_5_%284B%29_Vision.py)** — 손글씨 수식 이미지를 LaTeX로 변환하도록 시각·언어 모델을 LoRA로 미세 조정합니다.
- **[Qwen3_5_(4B)_Vision_GRPO.py](notebooks/unsloth/Qwen3_5_%284B%29_Vision_GRPO.py)** — MathVista 이미지 수학 문제를 대상으로 정답·응답 형식 보상을 정의하고 GRPO/GSPO를 적용합니다.
- **[Qwen3_(4B)-GRPO.py](notebooks/unsloth/Qwen3_%284B%29-GRPO.py)** — 수학 풀이 형식을 먼저 SFT로 익힌 뒤, 정답과 형식 보상을 이용한 GRPO로 추론 능력을 학습합니다.
- **[Qwen3_5_(4B)-DPO.py](notebooks/unsloth/Qwen3_5_%284B%29-DPO.py)** — UltraFeedback의 선호·비선호 응답 쌍으로 Qwen3.5-4B를 4비트 LoRA DPO 학습하는 실습입니다.
- **[Qwen3_(0.6B)-Knowledge_Distillation.py](notebooks/unsloth/Qwen3_%280.6B%29-Knowledge_Distillation.py)** — GKDTrainer로 교사 모델의 출력 분포를 작은 학생 모델에 전달하며, 기본 Qwen3 구성은 1.7B 교사와 0.6B 학생을 사용합니다.
- **[Qwen3_(4B)-Tool_Calling.py](notebooks/unsloth/Qwen3_%284B%29-Tool_Calling.py)** — 여러 턴의 함수 호출 대화로 도구 선택·인자 생성과 도구 결과를 활용한 응답을 미세 조정합니다.
- **[Qwen3_(4B)_Instruct-QAT.py](notebooks/unsloth/Qwen3_%284B%29_Instruct-QAT.py)** — int4 양자화를 고려한 QAT와 LoRA 미세 조정을 적용해 저정밀 모델 학습·저장 과정을 익힙니다.
- **[Qwen3_(0_6B)-Phone_Deployment.py](notebooks/unsloth/Qwen3_%280_6B%29-Phone_Deployment.py)** — Qwen3-0.6B를 휴대전화 배포용 QAT로 학습하고 ExecuTorch의 .pte 형식으로 내보내 CPU 추론을 확인합니다.

환경 준비와 추가 패키지는 [unsloth 실행 안내](notebooks/unsloth/README.md)를 참고합니다. DPO 노트북은 형식·구문·채팅 형식만 검증했으며 GPU 학습은 미실행입니다. Phone 노트북의 CPU 추론 확인과 별개로 휴대전화 실기기 검증은 필요합니다.

## physicalai

실물 로봇 없이 MuJoCo의 SO-101 로봇팔로 Physical AI 파이프라인을 장별로 실습합니다.
참고: [로봇 없이 시작하는 Physical AI 따라하기](https://wikidocs.net/book/21363), [예제 코드](https://github.com/ady95/physicalai_tutorial).

1. Agent 루프 → 2. MuJoCo 기초 → 3. FK·IK·Rule 기반 Pick and Place → 4. 카메라 인식 → 5. 강화학습(PPO)
→ 6. 모방학습(LeRobot ACT) → 7. VLA(SmolVLA) → 8. 일반화 시험 → 9. LLM Planner

설치·자원 요구와 공개 체크포인트 사용법은 [notebooks/physicalai/README.md](notebooks/physicalai/README.md)를 참고합니다.

## 코드 참고
- 실습 코드는 아래 코드를 참고하여 제작하였습니다.
  - LLMs from scratch
    - 코드: [https://github.com/rasbt/LLMs-from-scratch]
    - 강의: [https://www.youtube.com/watch?v=yAcWnfsZhzo]
  - 밑바닥부터 만들면서 배우는 LLM(llm from scratch 한글 번역본)
    - 코드: [https://github.com/rickiepark/llm-from-scratch]
    - 강의: [https://youtu.be/R80Gfde4cpg?list=PLJN246lAkhQhgakhcxz-5GwG_NYuJgSv1]
