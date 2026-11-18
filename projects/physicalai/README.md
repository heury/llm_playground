# SO-101 Pick and Place 시연 데이터 생성

MuJoCo로 SO-101 로봇팔을 시뮬레이션하고, 규칙 기반 Pick and Place 시연을
LeRobotDataset 형식으로 기록합니다.

| 파일 | 내용 |
| --- | --- |
| `so101_sim.py` | 장면(바닥·블록·상자·카메라), 로봇 제어, Inverse Kinematics |
| `record_demos.py` | 시연 실행과 데이터셋 기록 |
| `test_pick_place.py` | 시연 동작 테스트, 영상(mp4)·결과(JSON) 저장 |
| `test_color_pick.py` | 색깔 블록 4개를 하나씩 상자 귀퉁이에 옮기는 테스트, 영상·결과 저장 |
| `eval_policy.py` | 학습한 SmolVLA 정책을 빨강 단독·4색·다른 색 명령 조건에서 평가, 영상·결과 저장 |
| `test_astra_color_pick.py` | GPT-6.2 Astra가 장면 이미지에서 지정 색 블록 ID를 선택하고 IK로 옮기는 테스트 |

SO-101 모델은 처음 실행할 때 `third_party/mujoco_menagerie/`에 내려받습니다.

## 실행

`notebooks/physicalai/`의 uv 환경(`uv sync --locked`)을 사용합니다.

```bash
cd projects/physicalai
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python record_demos.py --episodes 50
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--episodes` | 50 | 실행할 episode 수 |
| `--root` | `outputs/physicalai/datasets/so101_pickplace_sim` (저장소 루트 기준) | 저장 폴더 |
| `--seed` | 0 | 블록 위치 난수 시드 |
| `--keep-failed` | 끔 | 실패한 시연도 저장 |
| `--overwrite` | 끔 | 저장 폴더가 있으면 지우고 새로 기록 |

## 동작 테스트와 영상

```bash
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_pick_place.py --trials 5
```

블록 위치를 바꿔 가며 실행하고 `outputs/physicalai/<시각>-pick-place-test/`에
회차별 영상 `trial_XX.mp4`, 이어 붙인 영상 `all_trials.mp4`, 결과 `results.json`을 저장합니다.
`--camera`로 `view`(기본, 대각선), `top`, `wrist_cam` 중에서 고를 수 있습니다.

```bash
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_color_pick.py --trials 3
```

빨강·초록·노랑·보라 블록을 바닥에 흩어 놓고 색깔별로 한 번씩 상자에 옮깁니다.
결과는 `outputs/physicalai/<시각>-color-pick-test/`에 저장하고, 영상 왼쪽 위에 현재 목표 색을 표시합니다.

## 학습한 정책 평가

```bash
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python eval_policy.py \
    --policy caspar/smolvla_so101_pickplace_sim_150_rtx5090_20261118 --episodes 20 --lang-episodes 10
```

조건 `red_only`(학습과 같은 장면), `distractors`(4색 장면, 명령 red), `lang_<색>`(4색 장면, 다른 색 명령)을
같은 블록 배치로 실행합니다. 결과는 `outputs/physicalai/<시각>-smolvla-multiblock-eval/`에
조건별 영상 `videos/<조건>.mp4`와 `results.json`으로 저장합니다. GPU를 권장합니다.

## Astra로 색상 지시 테스트

`test_astra_color_pick.py`는 4색 블록 장면 3개를 만들고, `gpt-6.2-astra`가
한국어 지시에 맞는 블록을 이미지에서 선택하도록 한다. 선택된 블록은 기존 IK 스킬로 옮긴다.
각 지시 전에 같은 배치로 초기화하며, 대상 선택 정확도와 이동 성공률을 따로 기록한다.
모델에는 색상 정답이나 시뮬레이터 좌표를 제공하지 않는다.

```bash
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_astra_color_pick.py prepare \
    --out ../../outputs/physicalai/<실행-ID>
# 생성한 익명 라벨 이미지와 planner-prompt.txt를 지정 모델에 입력한다.
# 응답을 planner-selections.json으로 저장한 다음 실행한다.
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_astra_color_pick.py execute \
    --out ../../outputs/physicalai/<실행-ID>
```

모델 호출 명령, 입력 범위와 판정 기준은
[재사용 시나리오](../../test_scenarios/physicalai/astra-color-pick.md)를 참고한다.

## 기록 내용 (20 fps)

- `observation.state`: 관절각 6개 (팔 5 + 집게 1)
- `observation.images.top`, `observation.images.wrist`: 320×240 영상
- `action`: 그 순간 구동기 6개에 명령한 목표 각도
