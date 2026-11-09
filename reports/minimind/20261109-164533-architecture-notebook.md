# MiniMind 모델 구조 수업 노트북 검증

- 생성 파일: [Chapter_4_Exercise_MiniMind_Architecture.ipynb](../../notebooks/Chapter_4_Exercise_MiniMind_Architecture.ipynb)
- 기준 노트북: `gpt/Chapter_4_Excercise_GPT.ipynb`.
- 구현: `minimind/model/model_minimind.py`. 모델 소스나 기존 노트북을 수정하지 않았다.
- 로그·환경 정보·재실행 스크립트: `outputs/minimind/20261109-164533-architecture-notebook/`의 `execution.log`, `status.json`, `validate_notebook.py`.

## 수업 구성

텍스트/토큰 ID → 임베딩 → RMSNorm → GQA/QK Norm/RoPE → SwiGLU FFN → 두 잔차 연결 → 블록 쌓기 → 최종 Norm → LM head → greedy 생성 순서로 구성했다. 원본 GPT와 MiniMind의 차이 표, 텐서 크기 관찰, 수동 조립 비교, 파라미터 수/가중치 공유 실습, 확인 문제 8개와 접을 수 있는 해설을 포함했다. 별도 학습이나 optimizer 업데이트는 하지 않는다.

## 실행과 환경

저장소 루트에서 실행:

```bash
minimind/.venv/bin/python outputs/minimind/20261109-164533-architecture-notebook/validate_notebook.py
```

스크립트의 작업 디렉토리는 `minimind/`다. 11개 코드 셀을 하나의 Python 네임스페이스에서 순서대로 compile/exec했다. 실제 Jupyter 커널 프로토콜 또는 IDE UI를 통해 실행하지는 않았다.

Python 3.12.15, PyTorch 2.14.1+cu130, Transformers 4.57.6, CPU float32, seed 123. 이미 설치된 환경을 사용했고 의존성 재설치 및 uv.lock과의 일치 여부 확인은 하지 않았다. 검증에 약 3.84초 소요했다.

모델 소스 SHA-256: `f9f160198ff37382e9deedb387eb67af0a7c239c447d1141e50c11a2badb9bde`.
토크나이저 파일별 SHA-256 및 전체 모델 설정은 `status.json`에 기록했다. 외부 데이터나 학습된 가중치는 사용하지 않았다. 로컬 6,400개 어휘 토크나이저, `Hello, I am` 프롬프트, 무작위 초기화 2층 hidden_size 32 Dense 모델을 사용했다.

## 결과와 판정 근거

11개 코드 셀 모두 통과했다.

- 입력 `[2,7]`, 임베딩 `[2,7,32]`, logits `[2,7,6400]` shape 확인.
- RMSNorm 수식과 실제 모듈 일치, 기본 RoPE 회전 전후 Q/K 길이 보존 확인.
- SwiGLU 수동 계산과 모듈 일치. FFN이 변경하지 않은 앞 토큰 출력을 유지함을 확인.
- 한 블록의 두 잔차를 수동 조립한 결과와 MiniMindBlock 출력 일치.
- 전체 블록 스택·최종 Norm·LM head 수동 계산이 실제 모델 logits 및 최종 hidden과 일치.
- 임베딩/LM head의 Parameter 객체 공유 확인. 두 블록의 Query 투영 가중치는 별도 객체임을 확인.
- 파라미터 공식과 실제 계산 일치: 블록당 9,296개, 고유 파라미터 합계 223,424개.
- 마지막 토큰 ID 변경에도 앞 위치 logits 유지: 전체 모델 인과성 확인.
- 수동 다음 토큰 교차 엔트로피와 모델 내부 손실 일치. 모든 고유 파라미터에 유한하고 0이 아닌 기울기 확인. 손실 8.787182은 무작위 모델의 값이며 품질 지표로 해석하지 않음.
- 직접 구현한 생성 루프, 기본 generate, KV cache generate의 5토큰 greedy 결과 일치.
- 노트북 JSON 구조, 셀 ID 중복 없음, 실행 번호/출력 비움, 모든 상대 링크 대상 존재 확인.

## 한계 및 미실행

학습자용 설정 변경 문제는 제안이며 자동 실행한 결과로 계산하지 않았다. 학습/optimizer 업데이트, 패딩, GPU, 혼합 정밀도, 분산 실행, 공개 가중치 및 언어 품질 평가는 수행하지 않았다. 텍스트 구조도와 접이식 해설의 IDE 렌더링은 확인하지 않았다. 출력 없는 배포용 노트북과 검증 로그를 구분해 보관했다.
