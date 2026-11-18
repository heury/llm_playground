# SmolVLA 3색 장면 색 지정 Pick and Place

빨강·초록·노랑 블록이 함께 있는 장면에서 정책이 **명령 문장의 색** 블록을 파란 상자에 옮기는지 확인한다.
3색 데이터(`caspar/so101_multicolor_sim` 또는 8장 노트북으로 기록한 같은 형식)로 학습한 SmolVLA 체크포인트용이다.

## 입력

- 정책: 처음 실행한 대상은 `caspar/smolvla_so101_multicolor_sim_rtx5090_20261118`, 리비전
  `17611a55590b66f2d03060f200be8daa258cbf5c`. 다른 체크포인트는 `--policy`, `--revision`으로 바꾼다.
- 학습 데이터(열린 고리 확인용): `caspar/so101_multicolor_sim`, 리비전 `1d47ca0df5c9b958acd005817fdb5bf6a1e42bd7`
  - 150 episode, 색별 50개, episode당 172 프레임, 20 fps
- 명령 문장: `Pick up the {red|green|yellow} cube and put it in the blue box.`

## 장면 조건

학습 데이터를 만든 책의 `ch08/record_multicolor.py`와 같은 조건이어야 한다.
`notebooks/physicalai/physicalai_common.py`가 이 조건을 따른다.

- 블록 3개(`cube`=빨강, `cube_green`, `cube_yellow`, 한 변 3 cm), 상자 `BOX_XY = (0.05, 0.22)`
- 배치 `pc.random_layout`: 반지름 0.17~0.27 m, ±0.7 rad 부채꼴, 블록 간격 7 cm 초과
- 카메라: top·wrist 240×320. 정책 입력 키는 `pc.SMOLVLA_KEYS`(camera1·camera2)
- 제한 시간 12초. 정책 실행은 `pc.run_policy_episode`

`projects/physicalai/eval_policy.py`는 보라를 포함한 4색, 간격 8 cm, 제한 15초 장면이므로 이 시나리오에 쓰지 않는다.

장면이 일치하는지 먼저 확인한다. 이 장면에서 `pc.scripted_pick_and_place(robot, color=..., demo=True)`를
`RecordingRobot`으로 실행하면 172 프레임이 나와야 한다.

## 절차

작업 디렉토리는 `notebooks/physicalai/`(`uv sync --locked` 완료, GPU 필요)이다.
`<ID>`는 `YYYYMMDD-HHMMSS-smolvla-multicolor-eval` 형식으로 정한다.

1. 닫힌 고리 평가: 배치 10개 × 세 색 = 30 episode. RTX 3080 Ti에서 약 2~3분 걸린다.
   ```bash
   MUJOCO_GL=egl .venv/bin/python ../../test_scenarios/physicalai/smolvla_multicolor_eval.py \
       --nb $(pwd) --layouts 10 --seed 2000 --out ../../outputs/physicalai/<ID>
   ```
   - 같은 배치의 세 명령에는 같은 noise seed(배치 번호)를 쓴다. 그래야 결과 차이가 문장에서만 온다.
   - 결과는 `results.json`, `videos/cmd_<색>.mp4`(fixed 카메라, 왼쪽 위에 명령 색과 결과 표시)에 저장된다.
2. 열린 고리 확인: 결과가 나쁠 때 평가 코드가 원인인지 가린다. 데이터셋 약 110 MB를 내려받는다.
   ```bash
   MUJOCO_GL=egl .venv/bin/python ../../test_scenarios/physicalai/smolvla_multicolor_openloop.py \
       $(pwd) ../../outputs/physicalai/<ID>/openloop
   ```
   - episode 0·1·2·75·76·77·147·148·149의 t = 0, 30, 60, 100 관측으로 50-step chunk를 예측한다.
   - 다음 세 값을 잰다.
     - 데이터 행동과의 오차
     - 문장의 색만 바꿨을 때 예측 변화
     - noise seed만 바꿨을 때 예측 변화
   - 모델과 리비전이 스크립트 안에 고정되어 있다. 다른 체크포인트는 스크립트 상단의 값을 바꾼다.
3. 보고서를 `reports/physicalai/<ID>.md`에 작성한다.

## 판정

episode마다 하나를 판정한다.

- 성공: 명령한 색 블록만 상자 안(`pc.is_in_box`: x·y 5 cm 이내, 높이 5 cm 미만)
- 오답: 다른 색 블록이 상자 안
- 실패: 12초 안에 아무것도 넣지 못함

보조 지표는 다음과 같다.

- 가장 가까이 간 블록: 그리퍼 끝과 수평 거리가 최소였던 블록
- 들어 올림: 블록 높이 5 cm 초과
- 명령과 무관한 배치 수: 같은 배치에서 세 명령 모두 같은 블록으로 간 배치 수

해석 기준은 다음과 같다.

- 언어를 따른다고 말하려면 성공률과 함께 "가장 가까이 간 블록 = 명령 색" 비율이 우연(1/3)보다 분명히 높아야 한다.
  예를 들어 Wilson 95% 하한이 1/3을 넘어야 한다.
- "명령과 무관한 배치 수"가 배치 수와 같으면 문장을 무시한다고 본다.
- 열린 고리에서 t≥30 오차가 수 cm·수 0.01 rad 수준으로 데이터를 따라가면 로딩·전처리 경로는 정상으로 본다.
  문장을 바꿨을 때의 변화가 noise를 바꿨을 때의 변화보다 작으면 색 단어가 행동에 거의 반영되지 않는다는 뜻이다.
- 30 episode로는 성공률 0일 때 95% 상한이 약 11%다. 더 좁은 구간이 필요하면 `--layouts`를 늘린다.

## 기록

| 실행 ID | 정책 | 성공 | 명령 색 블록으로 감 | 명령과 무관한 배치 | 문장/noise 변화(rad) | 보고서 |
|---|---|---:|---:|---:|---|---|
| 20261119-014842 | multicolor_sim_rtx5090_20261118 | 0/30 | 10/30 | 10/10 | 0.008 / 0.035 | [보고서](../../reports/physicalai/20261119-014842-smolvla-multicolor-eval.md) |
