# Physical AI 실습 노트북

실물 로봇 없이 MuJoCo 시뮬레이션으로 Physical AI 파이프라인 전체를 따라가는 marimo 노트북 모음입니다.
위키독스 책 [로봇 없이 시작하는 Physical AI 따라하기](https://wikidocs.net/book/21363)의 예제 코드를
장별 노트북으로 다시 구성했습니다.

```text
Agent 루프 → MuJoCo 세계 → FK·IK·Rule 기반 Pick and Place → 카메라 인식
→ 강화학습(PPO) → 모방학습(ACT) → VLA(SmolVLA) → 일반화 시험 → LLM Planner
```

## 노트북

| 장 | 노트북 | 내용 | 필요 자원 |
| --- | --- | --- | --- |
| 1 | [ch01_agent_basics.py](ch01_agent_basics.py) | 관측→판단→행동 세 줄 루프, GridWorld, 랜덤·규칙 Policy | CPU |
| 2 | [ch02_mujoco_basics.py](ch02_mujoco_basics.py) | 환경 확인, MJCF·`MjModel`/`MjData`, 낙하·공기저항, SO-101 관절 | CPU, Menagerie 다운로드(수십 MB) |
| 3 | [ch03_kinematics_pick_place.py](ch03_kinematics_pick_place.py) | Pose·quaternion, FK, Jacobian, DLS IK, Rule 기반 Pick and Place, 작업 영역 | CPU |
| 4 | [ch04_vision.py](ch04_vision.py) | OpenCV, HSV 색 검출, RGB-D 가상 카메라, 핀홀 역투영, Vision Pick and Place, (선택) YOLO | CPU, YOLO는 `--extra vision` |
| 5 | [ch05_reinforcement_learning.py](ch05_reinforcement_learning.py) | Gymnasium 환경, PPO, 로봇팔 reach, sparse·dense·shaped 보상 비교 | CPU, 30만 step 약 1.5분 |
| 6 | [ch06_imitation_act.py](ch06_imitation_act.py) | LeRobot 데이터셋, 시연 기록, `lerobot-train` ACT 학습·평가, action chunk 비교 | `--extra lerobot`, 학습은 NVIDIA GPU |
| 7 | [ch07_smolvla.py](ch07_smolvla.py) | SmolVLA 불러오기·평가, 파인튜닝 명령, 문장 변화 실험 | `--extra lerobot`, GPU, 다운로드 약 1.7 GB |
| 8 | [ch08_generalization.py](ch08_generalization.py) | 3색 블록 색 지정, 블록 위치 일반화, 조명·바닥·카메라·색 변화 | `--extra lerobot`, GPU |
| 9 | [ch09_llm_planner.py](ch09_llm_planner.py) | 로봇 도구 계층, 도구 호출 루프, 규칙 Planner·trace 재생·실제 LLM | CPU, 실제 LLM은 `--extra llm`과 API 키 |

`physicalai_common.py`는 여러 장이 공유하는 장면·로봇·IK·카메라·평가 도우미이며 실습 노트북이 아닙니다.
각 장에서 직접 만들어 본 코드(3장 Pick and Place 등)를 다음 장에서는 이 모듈의 같은 함수로 가져다 씁니다.

## 설치와 실행

Python 3.12 / Linux x86_64 기준입니다. 이 디렉토리에서 실행하세요.

```bash
cd notebooks/physicalai
uv sync --locked                    # 1~5장
uv sync --locked --extra lerobot    # 6~8장 (LeRobot, 수 GB)
uv sync --locked --all-extras       # 전체 (YOLO, LeRobot, OpenAI SDK)
uv run marimo edit ch01_agent_basics.py
```

marimo의 Python 환경은 `notebooks/physicalai/.venv/bin/python`입니다.
다른 가상환경이 활성화되어 있으면 `deactivate` 후 실행하세요. extra를 바꿨다면 marimo를 다시 시작하세요.

- 화면이 없는 Linux에서는 `physicalai_common.py`가 `MUJOCO_GL=egl`을 자동으로 정합니다. EGL이 없으면
  `sudo apt-get install libegl1`, GPU 없이 CPU 렌더링은 `libosmesa6` 설치 후 `MUJOCO_GL=osmesa uv run marimo edit ...`.
- SO-101 로봇 모델은 처음 실행할 때 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)에서
  `robotstudio_so101` 폴더만 `third_party/mujoco_menagerie/`로 내려받습니다(git 필요). 이미 받은 사본이 있으면 `MENAGERIE_DIR`로 지정하세요.
- 노트북을 열면 가벼운 셀(시뮬레이션·그림)은 바로 실행되고, 학습·다운로드·반복 평가는 **버튼을 눌러야** 실행됩니다.
- matplotlib 그림의 라벨은 한글 글꼴이 없는 환경을 고려해 영어로 썼습니다.

## GPU가 없거나 학습을 건너뛰려면

원본 저자의 [릴리스](https://github.com/ady95/physicalai_tutorial/releases)에 공개된 체크포인트를 노트북의 버튼으로 내려받아
평가에 바로 쓸 수 있습니다. 저장 위치는 `outputs/physicalai/checkpoints/`입니다.

| 이름 | 크기 | 쓰는 곳 | 학습 조건(체크포인트의 `train_config.json`) |
| --- | --- | --- | --- |
| `act_so101_v2_10k` | 약 190 MB | 6·8장 | ACT, 데이터셋 `so101_pickplace_sim_150`, 30,000 step, batch 16, 위 카메라 `top` |
| `smolvla_so101_v2_10k` | 약 720 MB | 7장 | SmolVLA, `so101_pickplace_sim_150`, 10,000 step, batch 8 |
| `smolvla_multicolor_v2_20k` | 약 720 MB | 8장 | SmolVLA, 3색 `so101_multicolor_sim_240`, 20,000 step, batch 8 |
| `ch09_traces` | 약 90 KB | 9장 | gpt-6-astra 실행 기록(대화 전문) |

## 원본과 달라진 점

- 원본은 장별 Python 스크립트(`chXX/*.py`)와 셸 스크립트입니다. 노트북은 같은 장면·상수·판정(`is_in_box`)을 쓰되,
  슬라이더로 값을 바꾸고 결과를 표·그림·영상으로 바로 보도록 재구성했습니다.
- 잠금 파일 기준 LeRobot 0.7, MuJoCo 4.0을 씁니다(원본 책은 LeRobot 0.6, MuJoCo 3.x). LeRobot 0.7의 `lerobot-train`은
  `--wandb.enable` 대신 `--trackio.enable=false`를 받습니다. 공개 체크포인트의 성공률은 책의 수치와 다를 수 있습니다.
- 5장 PPO는 프로세스 8개(`SubprocVecEnv`) 대신 한 프로세스의 `DummyVecEnv`로 학습합니다(노트북 셀에 정의한 환경을 그대로 쓰기 위해).
- 4장 YOLO 절은 COCO 사전학습 검출기만 다룹니다. 원본의 YOLOE visual prompt Pick and Place(`ch04/yolo_pick_and_place.py`)는 포함하지 않았습니다.
- 9장은 API 키 없이도 루프를 돌려 볼 수 있도록 규칙 기반 Planner(`scripted`)와 trace 재생(`replay`) 두뇌를 추가했습니다.
- 8장 환경 변화 실험에 블록 색 변화(`cube_blue`)와 4장 색 검출의 비교를 추가했습니다.

## 산출물

실행할 때마다 저장소 루트의 `outputs/physicalai/<날짜-시간-장>/`에 영상·그림·지표 JSON·학습 로그가 저장됩니다.
여러 장이 함께 쓰는 데이터셋은 `outputs/physicalai/datasets/`, 학습 결과와 공개 체크포인트는
`outputs/physicalai/checkpoints/`에 둡니다. 보고서는 `reports/physicalai/<실행 ID>.md`에 작성합니다.

## 출처와 라이선스

- 예제 코드 원본: [ady95/physicalai_tutorial](https://github.com/ady95/physicalai_tutorial) (MIT License).
  `physicalai_common.py`와 각 노트북의 장면·IK·로봇 래퍼·환경·도구 계층 코드는 원본을 옮기고 고친 것입니다.
- SO-101 모델: MuJoCo Menagerie `robotstudio_so101` (해당 폴더의 LICENSE를 따릅니다).
- LeRobot, SmolVLA: Hugging Face (Apache License 2.0). 공개 체크포인트는 원본 저자의 릴리스입니다.
