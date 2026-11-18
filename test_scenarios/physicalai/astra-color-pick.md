# Astra 카메라 기반 색상 지정 로봇 테스트

## 목적

빨강·초록·노랑·보라 블록을 함께 배치한 장면에서 `gpt-6.2-astra`가
한국어로 지시한 색의 블록을 이미지에서 선택하고, SO-101 로봇이 그 블록을
파란 상자에 옮기는지 검증한다.

LLM은 대상 ID를 선택하며, 실제 동작은 기존 `record_demos.pick_and_place`의
IK·집게 제어가 담당한다. SmolVLA 정책 학습/추론이나 LLM의 직접 관절 제어를
검증하는 시나리오는 아니다.

## 장면과 지시

- 기본 seed 62, 무작위 배치 3개. 각 배치에 4색 블록을 모두 포함한다.
- 블록은 로봇 앞 반지름 0.17~0.27m, 각도 ±0.7rad, 블록 간격 0.08m 이상에 놓는다.
- 카메라 이미지는 MuJoCo `top` 640×480.
- 각 블록에 A/B/C/D 라벨을 붙인다. 라벨-색상 매핑은 배치마다 무작위로 바뀐다.
- 각 배치에서 빨강·초록·노랑·보라를 각각 명령한다. 총 12개 배치/지시 쌍이다.
- 예: `초록색 블록만 파란 상자에 옮겨 주세요.`

## 모델 입력과 실행

작업 디렉토리: `projects/physicalai/`.
MuJoCo 렌더링에는 기존 `notebooks/physicalai/.venv` 환경을 사용한다.

```bash
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_astra_color_pick.py prepare \
  --out ../../outputs/physicalai/<실행-ID> --layouts 3 --seed 62
```

`planner-prompt.txt`, `selection-schema.json`과 `scene-00.png` 등 장면 이미지만
모델에 제공한다. `scenes.json`의 정답 매핑과 시뮬레이터 좌표는 모델에 제공하지 않는다.
준비 파일을 읽거나 도구를 호출하지 않고 이미지만으로 답하도록 요청한다.

기존 ChatGPT 인증을 쓸 경우 Codex CLI로 지정 모델을 호출한다.
다음의 `<실행경로>`는 실행 폴더의 절대 경로다.

```bash
codex exec --ephemeral --ignore-user-config --skip-git-repo-check \
  --sandbox read-only --model gpt-6.2-astra -C /tmp \
  --image <실행경로>/scene-00.png \
  --image <실행경로>/scene-01.png \
  --image <실행경로>/scene-02.png \
  --output-schema <실행경로>/selection-schema.json \
  --output-last-message <실행경로>/planner-selections.json --json - \
  < <실행경로>/planner-prompt.txt \
  > <실행경로>/planner-events.jsonl 2> <실행경로>/planner-stderr.log
```

응답은 전체 12개 scene/instruction 쌍에 대상 ID 또는 NONE을 포함한다.
실행 코드는 누락·중복·잘못된 ID를 검증한 뒤 물리 동작을 시작한다.

```bash
MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_astra_color_pick.py execute \
  --out ../../outputs/physicalai/<실행-ID> --seed 62
```

각 명령 전에 같은 배치로 초기화하고, 선택된 ID의 블록을 기존 IK 스킬에 전달한다.
모델이 잘못 선택하더라도 정답으로 바꾸지 않는다.

## 판정과 저장

- 대상 선택 정답: 모델의 ID가 지시한 색 블록과 일치.
- 이동 성공: 지시한 색만 상자에 들어가고 다른 색은 상자 안에 없음.
- 잘못된 이동: 다른 색이 상자에 들어감.
- 동작 실패: 아무 블록도 상자에 들어가지 않음.
- 상자 안 판정은 기존 `in_box`를 사용하며 x/y가 상자 중심에서 5cm 이내, z가 5cm 미만.
- episode마다 최종 위치·이동 거리·스킬 반환값·영상을 기록한다.

`outputs/physicalai/<실행-ID>/`에 전체 입력 이미지, 모델 응답과 이벤트,
`results.json`, `videos/`, `all_trials.mp4`를 저장한다.
`reports/physicalai/<실행-ID>.md`에 실행 환경·명령·결과·한계를 기록한다.
적은 수의 시뮬레이션 사례이며 실물 로봇 성능이나 새로운 물체에 대한 일반화를 뜻하지 않는다.
