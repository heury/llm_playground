# MiniMind Attention 실습 노트북 검증

- 생성 파일: [Chapter_3_Exercise_MiniMind_Attention.ipynb](../../notebooks/Chapter_3_Exercise_MiniMind_Attention.ipynb)
- 기준: `gpt/Chapter_3_Excercise_Attention.ipynb`의 Q/K/V, 인과 마스크, 단일·멀티헤드 실습 흐름.
- 사용 구현: `minimind/model/model_minimind.py`. 기존 소스는 수정하지 않았다.
- 산출물: `outputs/minimind/20261109-155136-attention-notebook/`의 `execution.log`, `status.json`, `validate_notebook.py`.

## 실행 및 환경

저장소 루트에서 실행:

```bash
minimind/.venv/bin/python outputs/minimind/20261109-155136-attention-notebook/validate_notebook.py
```

검증 스크립트는 작업 디렉토리를 `minimind/`로 설정하고, 단일 Python 네임스페이스에서 11개 코드 셀을 순서대로 `compile` / `exec`했다. Jupyter UI/커널 프로토콜을 통한 실행은 수행하지 않았다. Python 3.12.15, PyTorch 2.14.1+cu130, Transformers 4.57.6 환경에서 CPU float32로 실행했으며 성공한 실행 시간은 약 3.64초다. 설치된 PyTorch는 `minimind/pyproject.toml`의 `<3` 범위에 속하지만, 이 작업에서 의존성을 다시 설치하거나 잠금 파일과의 일치 여부를 확인하지 않았다.

모델 소스 SHA-256: `f9f160198ff37382e9deedb387eb67af0a7c239c447d1141e50c11a2badb9bde`.
가중치·데이터 리비전: 외부 모델/데이터를 사용하지 않음. seed 123과 노트북 내 합성 입력, 무작위 초기화 소형 모델을 사용했다.

## 결과와 판정 근거

11개 코드 셀 모두 통과했다.

- 단일 헤드 Attention 출력 shape와 수동 Q/K 정규화·RoPE·마스크·softmax·문맥 벡터 계산 결과가 일치했다.
- 어텐션 확률의 행 합은 1이며 미래 위치 확률은 0이었다.
- MHA/GQA/MQA 구성에서 출력 및 캐시 shape를 확인했고, GQA 수동 계산과 실제 출력이 일치했다.
- 마지막 입력 토큰을 바꿔도 앞선 토큰 출력은 변하지 않았다.
- 전체 Attention 입력과 토큰별 KV cache 출력/캐시가 일치했다.
- 같은 가중치의 명시적 Attention과 PyTorch SDPA 경로 출력이 일치했다.
- 실제 `MiniMindForCausalLM`에서 유한한 다음 토큰 손실과 Q/K/V/O 투영의 유한하고 0이 아닌 기울기를 확인했다.
- 전체 LM 순전파와 prefill + cache decode의 마지막 logits가 일치했다.

원본 노트북에는 출력 및 실행 번호를 저장하지 않았다. 생성 과정에서 발견한 문자열 줄바꿈 구문 오류는 수정 후 전체 셀을 다시 실행했다.

## 한계 및 미실행 항목

HTML 시각화 객체 생성까지 실행했으며 브라우저에서의 표 렌더링은 확인하지 않았다. GPU, 패딩 마스크, 혼합 정밀도, 성능 측정, 사전 학습 가중치, 실제 학습 및 언어 품질 평가는 수행하지 않았다. 마지막의 직접 바꿔 보는 실습은 학습자용 제안이며 통과한 검증으로 계산하지 않았다.
