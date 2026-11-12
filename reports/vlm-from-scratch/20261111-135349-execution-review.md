# VLM from Scratch 실행 검토

프로젝트 전용 가상 환경으로 12개 노트북의 실행 버튼을 활성화해 실행을 시도했다. 전체 학습 완료는 **0/12**다. 현재 호스트는 CUDA를 사용할 수 없고 필요한 가중치가 캐시에 없어 전체 실습을 완료할 수 없다. 이를 코드 정상 판정과 구분하고, 별도 CPU 소형 검증으로 실행 오류를 조사했다. 확인된 주요 문제는 **12부의 잘못된 언어 모델 임베딩 접근**과 **10부의 양자화 모델 크기 누락**이다. 검토 요청에 따라 노트북 소스는 수정하지 않았다.

- 실행 ID: `20261111-135349-execution-review` (Asia/Seoul)
- 작업 디렉토리: `notebooks/vlm-from-scratch/`; 검토 스크립트 호출은 저장소 루트에서 수행.
- 사용한 인터프리터: `/home/mind4u/dev/llm_playground/notebooks/vlm-from-scratch/.venv/bin/python`
- 산출물: [outputs/vlm-from-scratch/20261111-135349-execution-review/](../../outputs/vlm-from-scratch/20261111-135349-execution-review/)
- 재사용 절차: [테스트 시나리오](../../test_scenarios/vlm-from-scratch/README.md)

## 주요 발견

### 1. 높음: 12부의 수식 OCR 학습과 추론이 AttributeError로 실패

- 위치: [part-12-ocr-latex.py:323](../../notebooks/vlm-from-scratch/notebooks/part-12-ocr-latex.py#L323), [추론 경로:536](../../notebooks/vlm-from-scratch/notebooks/part-12-ocr-latex.py#L536).
- 두 경로 모두 `language_model.embeddings.word_embeddings(input_ids)`를 사용한다.
- 실제 SmolLM-135M의 설정을 확인하면 `model_type=llama`, 은닉 차원 576이다. `AutoModel`의 `LlamaModel`에는 `embeddings` 속성이 없다.
- 원본 클래스·함수를 추출해 실제 Transformers `ViTModel`/`LlamaModel`의 작은 설정과 연결했다. 비전 차원 768·언어 차원 576을 유지한 채 순전파와 `predict_latex()`를 각각 호출했고 모두 `AttributeError: 'LlamaModel' object has no attribute 'embeddings'`가 발생했다.
- 권장 수정: 두 곳을 `language_model.get_input_embeddings()(input_ids)`로 변경. 실제 가중치 없이도 재현되는 코드 오류다.
- 증거: `cpu-smoke-results.json`, `followup-results.json`의 traceback.

### 2. 중간: 10부의 양자화 모델 크기와 압축률이 부정확

- 위치: [part-10-compression.py:231](../../notebooks/vlm-from-scratch/notebooks/part-10-compression.py#L231).
- `get_model_size_mb()`는 `parameters()`와 `buffers()`만 합산한다. 동적 양자화 Linear의 packed 가중치는 이 합산에서 누락된다.
- 실제 `torch.quantization.quantize_dynamic`을 적용한 작은 HF 기반 모델에서 보고값은 **0.666 MiB**, 누락된 packed 가중치만 **1.846 MiB**였다. 두 값을 합한 텐서 용량은 적어도 **2.512 MiB**다. 이는 실제 배포 모델 용량이나 상주 메모리 전체를 측정한 수치가 아니다.
- INT8 순전파 자체는 통과했다. 문제는 크기·압축률 보고 방식이다.
- 권장 수정: 저장한 `state_dict`의 직렬화 크기를 FP32·INT8에서 같은 방식으로 측정하거나 packed 가중치를 명시적으로 합산하고 측정 기준을 표시한다.
- 증거: `followup-results.json`의 `quantized_size_packed_weights`.

### 3. 실습 해석 범위: 10부는 학습된 압축 모델의 품질 비교가 아님

정적 코드 확인 결과, 기준 모델은 이미지 캡션 체크포인트를 복구하지 않고 새 프로젝터·LM 헤드를 생성한다. `TinyVLM`도 기준 모델과 동일한 ViT-Base/SmolLM-135M을 사용하며 실제 증류 학습 호출은 생략되어 있다. 작은 모델 검증에서도 두 구조는 같은 크기였다. 따라서 이 노트북의 출력은 양자화·증류의 구현 시연으로 해석해야 하며, 학습된 작은 학생이 지식을 보존했다는 검증 결과로 해석할 수 없다. 이는 실행 예외와 구분한 소스 검토 결과다.

## 환경 및 실행 제약

- Linux / Python 3.12.15.
- torch 2.12.1, torchvision 0.27.1, transformers 4.57.6, marimo 0.25.2.
- datasets 5.0.1, diffusers 0.39.0, matplotlib 3.11.2.
- PyTorch CUDA 빌드 13.0, `torch.cuda.is_available() == False`. `nvidia-smi`는 NVIDIA 드라이버에 연결하지 못했다.
- 시작 시 남은 디스크 약 8.3GB. 필요한 ViT·SmolLM·Stable Diffusion·TrOCR 가중치는 Hugging Face 캐시에 없었다.
- 대용량 모델·데이터를 내려받지 않도록 앱 실행 시 `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 명시했다. 따라서 전체 실행에서 나온 캐시 부재 오류는 **오프라인 조건에서의 결과이며 인터넷 단절의 증거가 아니다**.
- 실제 네트워크 점검에서는 Hugging Face의 소형 config JSON 두 건을 성공적으로 읽었다. 가중치는 다운로드하지 않았다.

## 12개 앱 실행 시도

각 앱은 `app.run(defs={"mo": marimo, "run_notebook": SimpleNamespace(value=True)})`로 실행해 대기 버튼을 건너뛰었다. 모델·데이터 로딩·학습 셀을 비활성화하지 않았으며 최초 실패 지점에서 중단했다. 두 작업을 병렬로 실행하고 파일별 90초 제한을 적용했다. 실제 결과는 모두 타임아웃 전에 모델 로딩 단계에서 실패했다.

| 노트북 | 실행 결과 | 별도 소형·로컬 검증 |
| --- | --- | --- |
| `part-01-minimal-vlm.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·데이터 배치·1에포크 학습·체크포인트 재개 통과 |
| `part-02-object-detection.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·데이터 배치·JSON 파서·IoU 통과 |
| `part-03-visual-qa.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·데이터 배치 통과 |
| `part-04-multi-task.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·3종 데이터 배치 통과 |
| `part-05-multi-image.py` | 모델 로딩 실패(오프라인 캐시 없음) | 다중 이미지 순전파·역전파·생성·비교/선택 배치 통과 |
| `part-06-task-routing.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·태그 데이터 배치·작업 분류기 통과 |
| `part-07-chain-of-thought.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·CoT 데이터 배치 통과 |
| `part-08-referring-segmentation.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·데이터 배치·다각형 왕복 변환 통과 |
| `part-09-image-editing.py` | 모델 로딩 실패(오프라인 캐시 없음) | 순전파·역전파·생성·데이터 배치·PIL 연산 20종 통과 |
| `part-10-compression.py` | 모델 로딩 실패(오프라인 캐시 없음) | FP32 순전파·역전파·INT8 순전파 통과; 크기 누락 재현 |
| `part-11-image-generation.py` | 모델 로딩 실패(오프라인 캐시 없음) | 원형 마스크 생성 통과; 실제 확산 생성은 미실행 |
| `part-12-ocr-latex.py` | 모델 로딩 실패(오프라인 캐시 없음) | 수식 데이터 배치·LaTeX 6개 렌더링 통과; 모델 순전파·추론 실패 |

## CPU 검증 방법과 결과

`cpu_smoke.py`는 노트북 안의 원본 클래스·함수 정의를 AST로 추출해 실행한다. 다운로드·시연 셀 전체를 우회하므로 이 결과를 노트북 전체 실행 통과로 계산하지 않는다. 원본의 모델 계산 코드를 수정하지 않았다.

- 1~9부: 실제 HF ViT + LlamaForCausalLM, 은닉 차원 16, 층 1개, 어휘 64, 이미지 16×16. 2개 샘플의 유한 손실, 프로젝터 기울기, 고정된 비전 인코더, 2토큰 생성 검사.
- 데이터: 메모리에서 만든 PIL 이미지 6장과 캡션·객체·질문·답변·지칭 표현을 사용. WordLevel 기반 임시 토크나이저와 실제 ViTImageProcessor로 샘플·배치 생성, 응답 레이블 존재 및 패딩 마스킹 검사. 실제 Flickr8k/COCO/VQAv2/RefCOCO 형식 전체를 검증한 것은 아니다.
- 1부 학습: 원본 학습 함수로 소형 모델을 1에포크 학습하고 저장한 체크포인트에서 2에포크까지 재개. `smoke-caption.pt`는 **검증용 무작위 소형 모델**이며 실제 캡션 모델 가중치가 아니다.
- 10·12부: `AutoModel.from_pretrained` 로더만 무작위 초기화된 실제 ViTModel/LlamaModel 생성으로 대체. 원본의 고정 차원 768/576을 유지했다.
- 11부: 실제 Stable Diffusion·CLIP·BLIP 추론은 수행하지 않았다. 다운로드 없이 검사 가능한 마스크 함수만 실행했다.
- CPU 검증 총 57개 중 56개 통과, 1개 실패. 이 수에는 정의 로딩 검사 12개가 포함되며 모델 품질 통과율을 의미하지 않는다. 후속 검사에서 같은 12부 오류가 추론 경로에도 있음을 확인했다.
- PyTorch 동적 양자화 API에서 폐기 예정 경고가 있었으나 설치된 버전에서는 INT8 순전파가 통과했다.
- 프로젝트 가상 환경의 `marimo check --strict`: 종료 코드 0.

## 모델·데이터 리비전

실제 학습 가중치·데이터는 사용하지 않았다. 원본 소스의 SHA-256은 `source-sha256.json`, 설치 패키지는 `environment.json`에 기록했다. 원격 config 확인 결과는 다음과 같다.

- `HuggingFaceTB/SmolLM-135M`: `1d461723eec654e65efdc40cf49301c89c0c92f4`, 모델 유형 `llama`, 은닉 차원 576.
- `google/vit-base-patch16-224`: `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`, 모델 유형 `vit`, 은닉 차원 768.

이는 config 확인 리비전이며 실제 학습·추론에 사용한 가중치 리비전이 아니다. 합성 모델·데이터는 스크립트에 정의되어 있고 난수 시드는 0이다.

## 재현 명령

저장소 루트에서 실행한다. 검증 당시 소스는 수정하지 않았으며, 스크립트·로그·JSON 결과는 다음 경로에 함께 보관했다.

```bash
notebooks/vlm-from-scratch/.venv/bin/python outputs/vlm-from-scratch/20261111-135349-execution-review/run_notebooks.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 notebooks/vlm-from-scratch/.venv/bin/python outputs/vlm-from-scratch/20261111-135349-execution-review/cpu_smoke.py
notebooks/vlm-from-scratch/.venv/bin/python outputs/vlm-from-scratch/20261111-135349-execution-review/followup.py
notebooks/vlm-from-scratch/.venv/bin/marimo check --strict notebooks/vlm-from-scratch/notebooks/*.py
```

재실행은 이 실행 폴더의 결과를 갱신한다. 새 검증은 새 실행 폴더에 스크립트 사본을 두고 수행해야 한다. 원격 config·샘플 이미지 연결 결과는 `model-metadata.json`, `sample-image-network.json`을 참고한다.

## 미검증 범위와 권장 후속 작업

전체 12부 학습과 평가, 실제 모델 생성 품질, GPU 메모리 요구량, 실제 데이터셋의 모든 필드·좌표·마스킹 경계, 실제 SmolLM 토크나이저의 프롬프트 토큰 경계, Stable Diffusion의 실제 생성·평가 루프는 검증하지 못했다. 별도 GPU를 생성하거나 외부 서비스를 사용하지 않았다.

우선 12부의 두 임베딩 경로와 10부의 용량 측정을 수정한 뒤, NVIDIA 드라이버가 동작하고 필요한 모델·데이터를 저장할 여유가 있는 환경에서 실제 가중치·데이터로 각 파트의 축소 학습을 수행하는 순서가 적절하다.
