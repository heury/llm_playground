# Physical AI SO-101 시연 데이터 생성 코드 정리·Pick and Place 테스트 대화

저장일: 2026-11-19 (Asia/Seoul)

`projects/physicalai/`의 로봇 데이터 생성 코드 단순화, AGENTS.md 반영, 블록 옮기기·색깔별
블록 옮기기 시뮬레이션 테스트에 관한 사용자 요청과 어시스턴트 최종 답변을 보관합니다.
도구 호출·진행 설명과 IDE/환경 메타데이터는 제외했습니다. 소스·산출물 링크는 저장소 기준
상대 경로로 표기했습니다. 실제 실행 결과와 검증 근거는 연결된 보고서 및 outputs에 별도로 보존되어 있습니다.

## 사용자

projects/physicalai 아래 소스를 이용하여 robot 데이터를 생성할려고 해. 불필요한 부분은 제거해서 간단하고 이해하기 쉬운 코드로 수정해

## 어시스턴트

데이터 생성 코드를 파일 2개로 줄였습니다. 같은 seed로 실행했을 때 원래 코드와 똑같은 데이터가 나오는 것을 확인했습니다.

**새 구조** ([projects/physicalai/](../../projects/physicalai/))
- [so101_sim.py](../../projects/physicalai/so101_sim.py) (약 160줄): 장면 XML, Menagerie 모델 내려받기, IK, 로봇 제어를 담았습니다. 원래 `common/` 아래 파일 4개에 나뉘어 있던 내용입니다.
- [record_demos.py](../../projects/physicalai/record_demos.py) (약 130줄): Pick and Place 동작 순서와 LeRobotDataset 기록을 담당합니다.
- [README.md](../../projects/physicalai/README.md): 실행 방법과 기록되는 데이터 항목을 정리했습니다.

**삭제한 것**
- `ch03/`, `ch06/`, `common/` 폴더 전체. 아직 git에 커밋되지 않은 파일이라 git으로는 되돌릴 수 없습니다.
- `vision.py` 전체: 색 검출, depth, 픽셀→월드 좌표 변환은 데이터 생성에 쓰이지 않습니다. 이미지는 `mujoco.Renderer` 하나로 두 카메라를 번갈아 찍습니다.
- 다른 장에서만 쓰던 기능: 추가 블록, 조명·바닥 색 변경, 영상 저장, 시나리오 데모와 반복 실험 출력 등입니다.
- `--cube`, `--top-camera` 옵션과, 카메라 4대(`fixed`, `side`, `front`, `top_zoom`)

**바뀐 동작**
- 모델과 렌더러를 한 번만 만들고, episode마다 `reset(cube_xy)`로 블록 위치만 바꿉니다. 원래는 episode마다 장면 파일과 모델을 새로 만들었습니다.
- 기록은 `robot.on_step = record` 콜백으로 합니다. 원래는 로봇 클래스를 상속해서 기록했습니다.
- 기본 저장 경로가 AGENTS.md 규칙에 맞게 저장소 루트의 `outputs/physicalai/datasets/so101_pickplace_sim`으로 바뀌었습니다.

**확인 결과**
seed 0으로 3 episode를 돌렸을 때 원래 코드와 새 코드 모두 2개 성공, 1개 실패였습니다. 저장된 프레임 수는 각 172개, 합계 344개로 같았고, `observation.state` 합계도 154.4845로 같았습니다. 이미지 픽셀까지 비교하지는 않았습니다. 기존 파일을 지운 뒤 1 episode를 다시 돌려 정상 동작도 확인했습니다.

실행 방법:
```bash
cd projects/physicalai
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python record_demos.py --episodes 50
```

## 사용자

projects/physicalai도 agents.md에 추가해

## 어시스턴트

[AGENTS.md](../../AGENTS.md)에 `projects/physicalai`를 두 군데 추가했습니다.

- **전체 구조 트리**: `projects/` 아래에 `physicalai/`를 넣고 한 줄 설명을 붙였습니다.
- **새 섹션 "physicalai — SO-101 시연 데이터 생성"**: 「수정 후 확인할 항목」 바로 앞에 넣었습니다. 담은 내용은 다음과 같습니다.
  - 파일 구조와 읽는 순서
  - 실행 환경과 명령: 이 디렉토리에는 환경 파일이 없어 `notebooks/physicalai/.venv`를 씁니다.
  - 장면 XML이 `third_party/` 아래에 파일로 써지는 이유
  - 손목 카메라 `wrist_cam`은 장면 XML이 아니라 `so101.xml`에 정의되어 있다는 점
  - 기록 형식을 바꿀 때 `features`, `record()`, 학습 노트북을 함께 확인하라는 주의
  - 기본 저장 경로, `--overwrite` 동작, 문서만 보고 대량 기록을 실행하지 말라는 주의

`projects/physicalai/third_party/`(SO-101 모델 파일)는 지금 Git이 무시하지 않습니다. 그래서 문서에 "Git 제외"라고 적지 않았습니다. 커밋하지 않으려면 `.gitignore`에 추가해야 하는데, 원하시면 추가하겠습니다.

## 사용자

mujoco에서 로봇팔이 블럭을 집어서 접시에 옮기는 걸 테스트하고 결과를 영상으로 남겨

## 어시스턴트

MuJoCo에서 로봇팔이 블록을 집어 옮기는 테스트를 5번 돌렸고, 4번 성공했습니다(80%). 회차별 영상과 5회를 이어 붙인 영상을 저장했습니다. 장면에 접시는 없어서 기존의 파란 상자를 놓을 곳으로 썼습니다.

**결과** (블록 위치는 seed 0으로 무작위)

| trial | 블록 시작 위치 | 결과 |
|---|---|---|
| 1 | (+0.222, −0.074) | 성공 |
| 2 | (+0.136, −0.109) | 실패 |
| 3 | (+0.211, +0.137) | 성공 |
| 4 | (+0.219, +0.073) | 성공 |
| 5 | (+0.184, +0.128) | 성공 |

trial 2는 영상 프레임으로 원인을 확인했습니다. 블록이 로봇 바로 앞(약 0.17 m)에 있어서 팔을 많이 접은 자세로 잡았는데, 블록이 집게에 끼었습니다. 상자 위에서 집게를 벌려도 떨어지지 않았고, 팔이 다시 올라갈 때 함께 19 cm 높이까지 따라 올라갔습니다. 원인만 확인했고 고치지는 않았습니다.

**영상과 결과 파일** ([outputs/physicalai/20261119-012103-pick-place-test/](../../outputs/physicalai/20261119-012103-pick-place-test/))
- `all_trials.mp4`: 5회를 이어 붙인 영상 (대각선 카메라, 640×480, 30 fps)
- `trial_01.mp4` ~ `trial_05.mp4`: 회차별 영상, 각 약 9초
- `results.json`(회차별 시작·최종 위치와 성공 여부), `run.log`
- 보고서: [reports/physicalai/20261119-012103-pick-place-test.md](../../reports/physicalai/20261119-012103-pick-place-test.md)

**코드 변경**
- [test_pick_place.py](../../projects/physicalai/test_pick_place.py) 추가: `record_demos.py`의 Pick and Place 동작을 그대로 돌리면서 영상과 결과 JSON을 저장합니다. `--camera`로 `view`, `top`, `wrist_cam` 중 하나를 고를 수 있습니다.
- [so101_sim.py](../../projects/physicalai/so101_sim.py)에 영상용 대각선 카메라 `view`를 추가했습니다. 추가한 뒤에도 `record_demos.py`가 같은 seed에서 똑같은 데이터를 만드는 것(344 프레임, state 합계 154.4845)을 다시 확인했습니다.
- README와 AGENTS.md에 테스트 스크립트를 반영했습니다.

실제 접시 모양(낮은 원판)으로 바꾸길 원하시면 장면을 수정하겠습니다. 다만 장면이 바뀌면 생성되는 데이터셋도 함께 바뀝니다.

## 사용자

여러 색깔의 블럭을 바닥에 깔고 색깔별로 한번씩 가져가도록 테스트하고 결과를 영상으로 남겨

## 어시스턴트

색깔 블록 4개(빨강·초록·노랑·보라)를 바닥에 흩어 놓고 색깔별로 한 번씩 상자에 옮겼습니다. 배치 3가지에서 블록 12개가 모두 상자 안에 들어갔고(12/12), 결과를 영상으로 저장했습니다. 처음 실행에서는 2/12였는데, 로봇 동작 공통 코드(IK, 운반 높이)의 문제 두 가지를 고쳐서 얻은 결과입니다. 이 수정 때문에 `record_demos.py`가 만드는 데이터도 바뀝니다.

**결과** ([outputs/physicalai/20261119-013229-color-pick-test/](../../outputs/physicalai/20261119-013229-color-pick-test/))
- 회차마다 red → green → yellow → purple 순서로, 각각 상자의 서로 다른 귀퉁이에 옮깁니다. 세 회차 모두 4/4로 끝났습니다.
- 영상: `all_trials.mp4`(3회 이어 붙임), `trial_01~03.mp4`(각 약 35초). 영상 왼쪽 위에 현재 목표 색을 표시합니다.
- `results.json`에는 블록별로 시작 위치, 놓은 직후 성공 여부, 마지막까지 상자 안에 남았는지를 기록했습니다.
- 보고서: [reports/physicalai/20261119-013229-color-pick-test.md](../../reports/physicalai/20261119-013229-color-pick-test.md)
- 영상 프레임에서 블록 4개가 순서대로 상자 귀퉁이에 놓이는 것을 확인했습니다.

**테스트 중 발견한 문제와 수정**

| 실행 | 변경 | 결과 |
|---|---|---|
| 처음 | 없음 | 2/12 |
| 수정 1 | IK를 현재 자세와 대기 자세 두 곳에서 풀어 오차가 작은 쪽 선택 | 10/12 |
| 수정 2 | 운반 높이 `LIFT_Z` 0.12 → 0.10 m | 12/12 |

1. **IK가 관절 한계에 걸림:** 로봇에 가까운 상자 귀퉁이로 갈 때 IK가 관절 한계에 걸려 목표에서 16~20 cm 벗어났습니다. 다음 IK가 모두 그 자세에서 시작해서 뒤의 블록도 연달아 실패했습니다.
2. **운반 높이에 손이 닿지 않음:** 집게를 아래로 향한 채로는 로봇 쪽 상자 귀퉁이 위 12 cm에 어떤 시작 자세로도 닿지 않았습니다. 팔이 엉뚱한 자세로 간 뒤 3.3 rad를 한 번에 크게 움직이면서 블록을 던졌습니다.

**코드 변경**
- [so101_sim.py](../../projects/physicalai/so101_sim.py)
  - `SO101Sim(colors=[...])`로 블록 색을 고를 수 있습니다. 기본값은 이전과 같이 빨강 하나입니다.
  - IK가 대기 자세에서도 한 번 더 풉니다(수정 1).
- [record_demos.py](../../projects/physicalai/record_demos.py)
  - `pick_and_place(robot, color, place_offset)`가 색과 놓을 위치를 받습니다.
  - `in_box()`를 추가했고, `LIFT_Z`를 0.10으로 바꿨습니다.
- [test_color_pick.py](../../projects/physicalai/test_color_pick.py)를 추가했고, README·AGENTS.md·보고서를 갱신했습니다.

**데이터 생성에 미치는 영향:** IK와 `LIFT_Z`가 바뀌어서 `record_demos.py`가 이전과 같은 데이터를 만들지 않습니다.
- 단일 블록 성공률(30곳, seed 0): 이전 25/30 → 현재 30/30
- 앞서 돌린 `test_pick_place.py`(5회)도 이제 5/5입니다. 이전 보고서에는 4/5로 기록되어 있습니다.

수정 전에 실패한 두 실행의 산출물(`…012717…`, `…013027…`)은 지우지 않았고 보고서에 참조해 두었습니다.

## 사용자

이 대화 저장해

## 어시스턴트

대화 요약을 작업 기록으로 저장했습니다. 이후 요청에 따라 이 파일(`senarios/physicalai/`)로 옮겼습니다.

## 사용자

senarios/physicalai에 저장해

## 어시스턴트

이 파일에 저장했습니다.
