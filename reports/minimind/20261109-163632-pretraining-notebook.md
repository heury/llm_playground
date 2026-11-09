# MiniMind 사전 학습 노트북 검증

- 생성 파일: [Chapter_5_Exercise_MiniMind_Pretraining.ipynb](../../notebooks/Chapter_5_Exercise_MiniMind_Pretraining.ipynb)
- 참고 파일: `gpt/Chapter_5_Excercise_Pretraining.ipynb`.
- 사용 코드: `minimind/model/model_minimind.py`, `minimind/dataset/lm_dataset.py`. 기존 모델·데이터·학습 코드는 수정하지 않았다.
- 검증 로그·실행 스크립트: `outputs/minimind/20261109-163632-pretraining-notebook/`.
- 학습 데이터·캐시·곡선·샘플·체크포인트·manifest: `outputs/minimind/20261109-163648-pretraining-d417dc/`.

## 실행 명령과 환경

저장소 루트에서 다음 명령을 실행했다. 스크립트는 `minimind/`로 작업 디렉토리를 변경하고 11개 코드 셀을 동일 Python 네임스페이스에서 순서대로 compile/exec한다.

```bash
minimind/.venv/bin/python outputs/minimind/20261109-163632-pretraining-notebook/validate_notebook.py
```

Python 3.12.15, PyTorch 2.14.1+cu130, Transformers 4.57.6, datasets 5.0.1. CPU float32, 4 threads, seed 123. 설치된 환경을 사용했으며 의존성 재설치와 uv.lock 일치 검증은 하지 않았다. Jupyter UI/커널 프로토콜을 통한 실행은 수행하지 않았다.

모델 소스 SHA-256: `f9f160198ff37382e9deedb387eb67af0a7c239c447d1141e50c11a2badb9bde`.
데이터 코드 SHA-256: `1997c52c624b3a34f550ecc9e6875a183b1021438d04ce884bb28adcb496f9c1`.

외부 가중치·데이터 다운로드 없이 무작위 초기화한 1층 hidden_size 64 Dense MiniMind와 로컬 6,400개 어휘 BPE 토크나이저를 사용했다. 데이터는 노트북 안에서 작성한 서로 다른 합성 영어 문장 40개이며 36/4 문서로 분할했다. 데이터와 토크나이저 해시는 학습 산출물의 `manifest.json`에 기록했다.

## 결과

11개 코드 셀 모두 통과했다. 실제로 4에포크, 36 optimizer step을 실행했다.

| 항목 | 학습 전 | 학습 후 |
|---|---:|---:|
| 훈련 교차 엔트로피 | 8.760156 | 2.569735 |
| 검증 교차 엔트로피 | 8.743509 | 2.667551 |

다음 항목을 확인했다.

- 원문 문서의 훈련/검증 중복 없음, 토크나이저 encode/decode 일치.
- 모델 내부 정답 이동 및 PAD=-100 처리가 수동 교차 엔트로피와 일치.
- 모든 학습 손실과 평가 손실이 유한하며, gradient clipping에서 기울기의 유한성을 검사함.
- 학습 손실 감소. 검증 손실도 이번 실행에서는 감소함.
- 처리한 유효 다음 토큰 수 1656개가 데이터셋 기준 계산과 일치. PAD를 제외한 입력 토큰은 1800개이며 반복 에포크 포함.
- 저장 후 복원한 토크나이저 vocabulary 일치, 같은 입력 logits의 완전 일치, greedy 생성 결과 일치.
- `metrics.csv`, `loss.svg`, `samples.json`, `checkpoint.pt`, `config.json`, `tokenizer/`, `manifest.json`, `summary.json` 생성.

4에포크 후 생성 예시: `A students a a book every day.`. 문법과 반복 문제가 남아 있으며, 손실 감소를 실용적 언어 품질로 해석하지 않는다. 노트북 원본의 실행 번호·출력은 비워 두었다.

## 미실행 항목과 검증 한계

브라우저에서 SVG 렌더링은 확인하지 않았다. GPU, 혼합 정밀도, DDP, 기울기 누적, LR 스케줄, 대규모 데이터, 공개 가중치, 외부 품질 평가, 정확한 학습 중간 재개는 검증하지 않았다. 검증 문서는 훈련 문서와 같은 생성 패턴을 공유하므로 일반화 성능을 대표하지 않는다. 학습자용 추가 실습은 자동 검증 통과 항목으로 계산하지 않았다.
