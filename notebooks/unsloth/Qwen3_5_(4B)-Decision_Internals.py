import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 판단 모델 해부 실습 — Unsloth `FastDecisionModel` 내부 들여다보기

    [`Qwen3_5_(4B)-Decision.py`](Qwen3_5_%284B%29-Decision.py)는 `FastDecisionModel`의 공개 API로
    판단 모델을 학습하는 과정을 그대로 따라갑니다. 이 노트북은 그 API 아래에서 실제로 무슨 일이
    일어나는지 단계별로 열어 봅니다. 참고한 Unsloth 원본 파일은 다음과 같습니다.

    | 파일 | 역할 |
    | --- | --- |
    | `unsloth/models/decision_from_lm.py` | 일반 언어 모델(LM)에 새 판단 헤드를 붙여 Clef 형식 모델을 만듭니다. 헤드 크기 결정, 백본 고정·해제를 담당합니다. |
    | `unsloth/models/decision.py` | `FastDecisionModel`, `DecisionTrainer`, 손실 함수, 데이터 묶음(collator), 추론·보정·저장을 담당합니다. |
    | `unsloth/models/decision_datasets.py` | 공개 데이터셋을 판단 형식으로 변환하고 스키마 증강·혼합·오염 제거를 담당합니다. |
    | `unsloth/_vendor/clef/joint_schema_model.py` | Cloudflare Clef의 원본 구현입니다. 프롬프트 인코딩과 `JointSchemaHead`가 들어 있습니다. |

    ## 학습 목표

    1. 판단 모델이 텍스트 생성 모델과 어떻게 다른지 설명할 수 있다.
    2. `state` · `questions` · `gold` 데이터 형식과 세 질문 유형(`choice`, `noul`, `score`)을 다룰 수 있다.
    3. 하나의 프롬프트에 모든 질문을 넣고 **질문·선택지 토큰 구간(span)** 을 기록하는 인코딩 방식을 이해한다.
    4. `JointSchemaHead`가 은닉 상태에서 선택지별 점수(logit)를 만드는 과정을 텐서 shape 단위로 따라간다.
    5. 손실 함수(교차 엔트로피·라벨 스무딩·Brier·순서 거리)와 온도 보정(ECE)을 직접 계산해 본다.
    6. 작은 모델로 **헤드만 학습 → 백본까지 학습** 하는 2단계 레시피를 CPU에서 실행해 본다.
    7. 같은 과정을 Unsloth API로 GPU에서 실행한다.

    ## 실행 방법

    `notebooks/unsloth/`에서 실행합니다.

    ```bash
    uv sync --locked
    uv run --locked marimo edit 'Qwen3_5_(4B)-Decision_Internals.py'
    ```

    - **1~8부**는 GPU 없이 CPU에서 동작합니다. 처음 열면 토크나이저(수 MB)만 내려받습니다.
      CPU 미니 학습은 **미니 학습 실행** 버튼을 눌러야 시작하며, 약 1분 걸립니다.
    - **9부**는 NVIDIA GPU가 필요하며 **GPU 실습 실행** 버튼을 눌러야 시작합니다.
    - 실행 산출물은 저장소 루트의 `outputs/unsloth/<실행 ID>/`에 저장됩니다.
    """)
    return


@app.cell
def _():
    import html
    import importlib.metadata
    import importlib.util
    import json
    import math
    import random
    import sys
    import time
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import marimo as mo
    import torch
    import torch.nn.functional as F

    # Unsloth는 transformers보다 먼저 import해야 최적화 패치가 모두 적용됩니다.
    # GPU가 있으면 여기서 먼저 불러오고, CPU만 있으면 건너뜁니다(1~8부는 unsloth import 없이 동작).
    if torch.cuda.is_available():
        try:
            import unsloth  # noqa: F401

            unsloth_status = "GPU 감지: unsloth를 먼저 import했습니다."
        except Exception as _error:  # 드라이버·패키지 문제를 노트북 실행 전체의 실패로 만들지 않습니다.
            unsloth_status = f"unsloth import 실패({type(_error).__name__}): 9부는 실행할 수 없습니다."
    else:
        unsloth_status = "GPU 없음: 1~8부만 실행합니다."
    REPO_ROOT = Path(__file__).resolve().parents[2]


    def new_run_dir(tag):
        """저장소 루트의 outputs/unsloth/<날짜-시간-임의ID-작업명>/ 폴더를 만듭니다."""
        run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-{tag}"
        path = REPO_ROOT / "outputs" / "unsloth" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path


    print(unsloth_status)
    return (
        F,
        Path,
        html,
        importlib,
        json,
        math,
        mo,
        new_run_dir,
        random,
        sys,
        time,
        torch,
        unsloth_status,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Unsloth 원본 코드 불러오기

    `import unsloth`는 GPU를 요구하고 transformers 등을 패치합니다. CPU에서도 원본 로직을 그대로
    실습하기 위해, 설치된 패키지에서 **필요한 파일만 경로로 불러옵니다.** 아래 네 파일은 최상위에서
    표준 라이브러리와 `torch`만 import하므로 패키지 초기화 없이 실행됩니다.

    `decision.py`는 Unsloth 내부 모듈을 최상위에서 import하므로 이 방식으로 불러올 수 없습니다.
    그 안의 손실 함수와 추론 함수는 5부와 8부에서 짧게 옮겨 와 사용하고, 9부에서 실제 API로 실행합니다.
    """)
    return


@app.cell
def _(Path, importlib, sys, unsloth_status):
    _ = unsloth_status  # unsloth import(가능한 경우) 뒤에 실행되도록 순서를 고정합니다.
    UNSLOTH_DIR = Path(importlib.util.find_spec("unsloth").submodule_search_locations[0])
    UNSLOTH_VERSION = importlib.metadata.version("unsloth")


    def load_unsloth_source(relative_path):
        """unsloth 패키지를 초기화하지 않고, 설치된 원본 파일 하나만 모듈로 불러옵니다."""
        path = UNSLOTH_DIR / relative_path
        name = "unsloth_source_" + path.stem
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclass가 자신의 모듈을 찾을 수 있도록 먼저 등록합니다.
        spec.loader.exec_module(module)
        return module


    clef_src = load_unsloth_source("_vendor/clef/joint_schema_model.py")
    datasets_src = load_unsloth_source("models/decision_datasets.py")
    from_lm_src = load_unsloth_source("models/decision_from_lm.py")
    common_src = load_unsloth_source("models/_decision_common.py")
    print(f"unsloth {UNSLOTH_VERSION}: {UNSLOTH_DIR}")
    return (
        UNSLOTH_DIR,
        UNSLOTH_VERSION,
        clef_src,
        common_src,
        datasets_src,
        from_lm_src,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 생성 모델과 판단 모델

    텍스트 생성 모델은 다음 토큰을 하나씩 샘플링해 답을 **문장으로** 씁니다. 출력이 정해진 선택지 밖으로
    벗어날 수 있고, "얼마나 확신하는가"를 얻으려면 별도 처리가 필요합니다.

    판단 모델은 생성하지 않습니다. 입력과 질문을 **한 번의 순전파(prefill)** 로 읽은 뒤, 질문마다
    **허용된 선택지 각각에 점수(logit) 하나**를 냅니다. softmax를 거치면 선택지 확률이 되고,
    가장 높은 선택지가 답입니다. 출력은 항상 선택지 중 하나이며 확률이 함께 나옵니다.

    ```text
    [state + 질문1 + 선택지들 + 질문2 + 선택지들 + ...] ──LM 백본──▶ 은닉 상태 [L, H]
                                                           │
                                               JointSchemaHead (판단 헤드)
                                                           ▼
                                질문1: [z_a, z_b, z_c]   질문2: [z_true, z_false]   ...
    ```

    질문 유형은 세 가지입니다.

    | 유형 | 선택지 | 예 | 답 형식 |
    | --- | --- | --- | --- |
    | `choice` | `criteria` dict의 키 (**알파벳순 정렬**) | 담당 팀 | 선택지 이름 |
    | `noul` | 항상 `true`, `false` 두 개 | 환불 요청 여부 | `True` / `False` |
    | `score` | `criteria` 리스트의 인덱스 `0, 1, 2, ...` | 긴급도 | 등급 번호 |

    `question_options`가 질문을 선택지 목록으로 바꾸는 규칙을 직접 확인해 봅니다. `choice` 선택지는
    입력한 순서가 아니라 **정렬된 순서**로 바뀌는 점에 주의하세요.
    """)
    return


@app.cell
def _(clef_src, mo):
    TOY_QUESTIONS = {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "technical": "bugs, outages, errors",
                "billing": "invoices, payments, refunds",
                "sales": "pricing, new plans",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "today"],
        },
    }
    mo.ui.table(
        [
            {"질문": _name, "유형": _q["type"], "선택지 순서": [key for key, _ in clef_src.question_options(_q)]}
            for _name, _q in TOY_QUESTIONS.items()
        ],
        selection=None,
    )
    return (TOY_QUESTIONS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 확률을 답으로 바꾸기: `systemone_answer`

    선택지 확률이 나오면 `systemone_answer`가 Decision API 응답 형식으로 바꿉니다. `noul`은 `true`의
    확률, `choice`는 최고 확률 선택지와 신뢰도, `score`는 **등급의 기댓값**(`Σ 등급 × 확률`)을 냅니다.
    Unsloth의 `FastDecisionModel.predict`는 여기에 `answer`(최빈 선택지)를 덧붙입니다.

    슬라이더로 `score` 질문의 확률을 바꿔 보세요. `score`(기댓값)와 `answer`(가장 높은 등급)가 달라지는
    경우가 있습니다. 예를 들어 `[0.45, 0.10, 0.45]`이면 기댓값은 1.0이지만 가장 높은 등급은 0입니다.
    """)
    return


@app.cell
def _(mo):
    score_probability_sliders = mo.ui.array(
        [mo.ui.slider(0.0, 1.0, step=0.05, value=_v, label=f"등급 {_i}") for _i, _v in enumerate([0.45, 0.10, 0.45])]
    )
    score_probability_sliders
    return (score_probability_sliders,)


@app.cell
def _(TOY_QUESTIONS, clef_src, common_src, json, mo, score_probability_sliders):
    _raw = [max(_v, 1e-6) for _v in score_probability_sliders.value]
    _probs = {str(_i): _v / sum(_raw) for _i, _v in enumerate(_raw)}
    _answer = clef_src.systemone_answer(TOY_QUESTIONS["urgency"], _probs)
    _predicted = common_src._predicted(TOY_QUESTIONS["urgency"], _answer, _probs)
    _noul = clef_src.systemone_answer(TOY_QUESTIONS["refund"], {"true": 0.83, "false": 0.17})
    mo.md(
        f"""
    **score 응답** (`systemone_answer` + `_predicted`)
    ```json
    {json.dumps(_predicted, indent=2, ensure_ascii=False)}
    ```
    **noul 응답 예** (`true` 확률 0.83)
    ```json
    {json.dumps(_noul, ensure_ascii=False)}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 데이터 형식: `state`, `questions`, `gold`

    학습 데이터의 한 행(row)은 입력 `state`, 질문 dict `questions`, 정답 dict `gold`로 구성됩니다.
    `state`는 문자열이나 임의의 JSON 값이 될 수 있습니다. 한 행에 질문이 여러 개 있으면 **한 번의
    순전파로 함께 판단**합니다.

    `gold`는 정답 라벨 하나일 수도 있고, 여러 사람의 판단이 갈린 경우처럼 **확률 분포(soft label)** 일
    수도 있습니다. `_target_for`(`_decision_common.py`)가 이를 선택지 순서에 맞춘 목표 분포
    `target`과 정답 인덱스 `label`로 바꿉니다. 아래 예에서 `gold`의 형태별 변환 결과를 확인하세요.
    """)
    return


@app.cell
def _(TOY_QUESTIONS, clef_src, common_src, mo):
    _cases = [
        ("choice, 라벨 하나", "team", "billing"),
        ("choice, 확률 분포", "team", {"probabilities": {"billing": 0.7, "sales": 0.3}}),
        ("noul, bool", "refund", True),
        ("noul, true 확률", "refund", {"noul": 0.8}),
        ("score, 등급 번호", "urgency", 2),
        ("score, 확률 분포", "urgency", {"probabilities": {"1": 0.4, "2": 0.6}}),
    ]
    _rows = []
    for _desc, _qid, _gold in _cases:
        _question = TOY_QUESTIONS[_qid]
        _keys = [key for key, _ in clef_src.question_options(_question)]
        _target, _label = common_src._target_for(_question["type"], _keys, _gold)
        _rows.append(
            {"경우": _desc, "gold": str(_gold), "선택지": _keys, "target": [round(_t, 2) for _t in _target], "label": _label}
        )
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 공개 데이터 변환과 스키마 증강 (`decision_datasets.py`)

    `decision_datasets.py`의 변환기(converter)는 분류·NLI·객관식·도구 선택 데이터셋의 행을 판단 형식으로
    바꿉니다. 변환기는 일반 dict 행을 받으므로 **데이터를 내려받지 않고도** 동작을 확인할 수 있습니다.
    아래는 BoolQ(예/아니요), SST-5(5단계 감성), AG News(주제 분류) 형식의 직접 만든 행을 변환한 결과입니다.

    변환된 행에는 학습에 쓰이지 않는 `_variants`가 붙습니다. 질문 문장의 바꿔 쓰기(paraphrase)와 대체
    질문 ID 후보로, 다음 단계의 증강에서 사용합니다.
    """)
    return


@app.cell
def _(datasets_src, json, mo):
    converted_rows = [
        *datasets_src.convert_boolq(
            [{"passage": "The Nile flows north into the Mediterranean Sea.", "question": "does the nile flow north", "answer": True}]
        ),
        *datasets_src.convert_sst5([{"text": "A warm, funny film with a weak ending.", "label": 3}]),
        *datasets_src.convert_ag_news([{"text": "The central bank raised interest rates by 0.25%.", "label": 2}], None),
    ]
    mo.md("```json\n" + json.dumps(converted_rows, indent=2, ensure_ascii=False) + "\n```")
    return (converted_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 스키마 증강: `augment_row`

    판단 모델은 학습 때 본 질문 ID·선택지 이름만 외우면 안 됩니다. 실제 사용자는 처음 보는 스키마를
    보내기 때문입니다. `augment_row`는 같은 행을 다음과 같이 무작위로 변형합니다(`AugmentConfig` 기본값).

    | 변형 | 확률 | 내용 |
    | --- | --- | --- |
    | `rename_question_ids` | 0.3 | 질문 ID를 대체 ID나 `q1`, `field_a` 같은 의미 없는 ID로 교체 |
    | `rename_options` | 0.3 | 선택지 이름을 `A, B, C`나 `abc3` 같은 코드로 교체 (의미는 설명으로 이동, `gold`도 함께 변환) |
    | `paraphrase_instructions` | 0.4 | 질문 문장을 `_variants`의 다른 표현으로 교체 |
    | `drop_instructions` | 0.1 | 질문 문장을 제거 (ID가 유일한 설명이 되므로 ID는 유지) |
    | `derived_questions` | 0.5 | `choice` → "답이 X인가?", `score` → "최소 X 이상인가?" 같은 `noul` 질문 추가 |
    | `state_as_text` | 0.3 | dict `state`를 `Key: value` 줄글로 변환 |

    씨앗(seed)을 바꿔 가며 같은 AG News 행이 어떻게 변하는지 보세요. `gold`가 바뀐 선택지 이름을 따라가는지도 확인하세요.
    """)
    return


@app.cell
def _(mo):
    augment_seed = mo.ui.slider(0, 30, value=0, label="증강 seed", show_value=True)
    augment_seed
    return (augment_seed,)


@app.cell
def _(augment_seed, converted_rows, datasets_src, json, mo, random):
    _augmented = datasets_src.augment_row(converted_rows[2], random.Random(augment_seed.value))
    _augmented.pop("source", None)
    mo.md("```json\n" + json.dumps(_augmented, indent=2, ensure_ascii=False) + "\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 데이터 혼합과 오염 제거

    `build_decision_mixture(sources, n_rows)`는 여러 공개 데이터(`SOURCES`에 등록된 14개)를 가중치대로
    섞고, 행마다 새로 증강합니다. 원본이 부족하면 복원 추출하며, 반복된 행도 증강이 달라집니다.
    평가 전용 벤치마크(`EVAL_ONLY_SOURCES`)는 학습에 넣으려 하면 오류를 냅니다.

    `Decontaminator`는 평가 텍스트와 **연속 13단어(shingle)** 가 하나라도 겹치는 학습 행을 제거합니다.
    아래에서 n을 작게 해 동작을 확인합니다. 실제 혼합은 데이터 다운로드가 필요하므로 9부에서 선택적으로 사용합니다.
    """)
    return


@app.cell
def _(datasets_src, mo):
    _cleaner = datasets_src.Decontaminator(["The central bank raised interest rates again"], n=4)
    _checks = [
        {"state": "The central bank raised interest rates by 0.25%."},
        {"state": "Rates were left unchanged by the central bank."},
    ]
    mo.vstack(
        [
            mo.ui.table(
                [{"source": _name, "hf_id": _s.hf_id, "license": _s.license} for _name, _s in datasets_src.SOURCES.items()],
                selection=None,
                page_size=5,
            ),
            mo.ui.table(
                [{"학습 행 state": _row["state"], "오염(제거 대상)": _cleaner.contaminated(_row)} for _row in _checks],
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 프롬프트 인코딩: `encode_record`

    판단 모델의 입력은 Qwen 채팅 템플릿 형식의 **하나의 프롬프트**입니다. `encode_record`는 텍스트를
    만드는 동시에 각 질문 문장과 각 선택지가 몇 번째 토큰부터 몇 번째 토큰까지인지(**span**) 기록합니다.
    헤드는 이 구간의 은닉 상태를 평균해 질문 벡터·선택지 벡터를 만듭니다.

    - `state`가 길면 **state의 뒷부분만 잘립니다.** 질문과 선택지는 잘리지 않습니다.
    - 마지막은 `JOINT SCHEMA DECISIONS:`로 끝납니다. 이 마지막 토큰의 은닉 상태가 전체 요약(global vector)입니다.
    - `<think>\n\n</think>`는 Qwen3.5의 사고(thinking) 블록을 비워 둔 것입니다.

    입력을 고쳐 보세요. 색으로 표시한 구간이 헤드가 읽는 부분입니다.
    <span style="background:rgba(59,130,246,.30)">질문 문장</span>,
    <span style="background:rgba(249,115,22,.30)">선택지</span>,
    <span style="background:rgba(16,185,129,.30)">마지막 토큰</span>.
    """)
    return


@app.cell
def _(mo):
    from transformers import AutoTokenizer

    # Qwen3.5 0.8B·2B·4B는 같은 토크나이저를 사용합니다(어휘 248,077개).
    TOKENIZER_NAME = "unsloth/Qwen3.5-0.8B"
    with mo.status.spinner(title=f"{TOKENIZER_NAME} 토크나이저를 불러오는 중..."):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state_input = mo.ui.text_area(
        value="Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
        label="state",
        full_width=True,
    )
    max_length_input = mo.ui.slider(200, 400, step=10, value=400, label="max_length (토큰)", show_value=True)
    mo.vstack([state_input, max_length_input])
    return TOKENIZER_NAME, max_length_input, state_input, tokenizer


@app.cell
def _(TOY_QUESTIONS, clef_src, html, max_length_input, mo, state_input, tokenizer):
    try:
        preview_record = clef_src.encode_record(
            tokenizer, {"state": state_input.value, "questions": TOY_QUESTIONS}, max_length=max_length_input.value
        )
    except ValueError as _error:
        preview_record = None
        _view = mo.callout(f"인코딩 실패: {_error}", kind="danger")
    if preview_record is not None:
        _color = {}
        for _q in preview_record.questions:
            for _t in range(*_q.question_span):
                _color[_t] = "rgba(59,130,246,.30)"
            for _start, _end in _q.option_spans:
                for _t in range(_start, _end):
                    _color[_t] = "rgba(249,115,22,.30)"
        _color[len(preview_record.input_ids) - 1] = "rgba(16,185,129,.45)"
        _pieces = []
        for _t, _id in enumerate(preview_record.input_ids):
            _text = html.escape(tokenizer.decode([_id]))
            _pieces.append(f'<span style="background:{_color[_t]}">{_text}</span>' if _t in _color else _text)
        _spans = [
            {
                "질문": _q.question_id,
                "유형 번호": _q.question_type,
                "question_span": _q.question_span,
                "선택지": list(_q.option_ids),
                "option_spans": list(_q.option_spans),
            }
            for _q in preview_record.questions
        ]
        _view = mo.vstack(
            [
                mo.md(f"전체 토큰 수: **{len(preview_record.input_ids)}**"),
                mo.Html(
                    '<pre style="white-space:pre-wrap;font-size:12px;line-height:1.5">' + "".join(_pieces) + "</pre>"
                ),
                mo.ui.table(_spans, selection=None),
            ]
        )
    _view
    return (preview_record,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `max_length`를 줄이면 state 끝부분이 먼저 잘리고, 질문과 선택지만으로도 길이를 넘으면
    `schema requires ... tokens before state` 오류가 납니다. 질문 유형 번호는 Clef 기준으로
    `noul=0, choice=1, score=2`입니다.

    ## 5. 판단 헤드: `decision_from_lm.py`와 `JointSchemaHead`

    `FastDecisionModel.from_pretrained`에 일반 LM 이름을 주면 `decision_from_lm.load_lm_as_decision_model`이
    호출됩니다. 이 함수가 하는 일은 다음과 같습니다.

    1. 백본 LM을 불러옵니다(GPU에서는 Unsloth `FastModel`, 4비트 가능).
    2. 백본의 `hidden_size`로 `default_head_config`를 계산해 **새 `JointSchemaHead`** 를 만듭니다.
       (`head_init`을 주면 공개된 Clef 헤드로 시작할 수도 있습니다.)
    3. 헤드는 항상 **float32** 로 두고 `ClefDecisionModel(encoder=백본, head=헤드)`로 묶습니다.
    4. `decision_config`에 최대 길이, 질문 유형별 온도(처음엔 1.0), 원본 모델 이름을 기록합니다.

    헤드 크기는 백본 크기에 따라 정해집니다. `hidden_size ≥ 3072`이면 폭(width) 1024, 그보다 작으면 512입니다.
    아래 표의 매개변수 수는 meta 장치에서 헤드를 만들어 계산합니다(메모리를 쓰지 않습니다).
    """)
    return


@app.cell
def _(clef_src, from_lm_src, mo, torch):
    _rows = []
    for _hidden in (1024, 2048, 2560, 4096, 5120):
        _config = from_lm_src.default_head_config(_hidden)
        with torch.device("meta"):
            _count = sum(_p.numel() for _p in clef_src.JointSchemaHead(**_config).parameters())
        _rows.append({**_config, "헤드 매개변수(M)": round(_count / 1e6, 1)})
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 작은 실습용 모델 만들기

    CPU에서 구조를 따라가기 위해 Qwen3 구조의 **무작위 초기화** 백본(은닉 128, 2층)을 만들고, 같은
    `default_head_config`(폭 128로 지정)로 원본 `JointSchemaHead`를 붙입니다. 어휘는 Qwen3.5 토크나이저와
    같으므로 4부의 인코딩을 그대로 사용합니다.

    `ToyClefDecisionModel`은 `decision.py`의 `ClefDecisionModel.forward`를 줄여 옮긴 것입니다.
    백본의 마지막 은닉 상태를 헤드에 넘기고, 질문마다 길이가 다른 logit을 `-1e4`로 채워 `[질문 수, 최대 선택지 수]`
    텐서로 만듭니다. 출력 임베딩 가중치는 `.detach()`해서 넘기는데, 선택지 단어 자체의 의미(lexical)를
    읽는 용도일 뿐 이 경로로는 임베딩을 학습하지 않기 때문입니다.
    """)
    return


@app.cell
def _(clef_src, from_lm_src, torch):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    TOY_HIDDEN = 128


    class ToyClefDecisionModel(torch.nn.Module):
        """decision.py의 ClefDecisionModel을 CPU 실습용으로 줄인 것."""

        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder  # Unsloth와 같은 이름: freeze_backbone이 model.encoder를 찾습니다.
            self.head = head

        def forward(self, input_ids, attention_mask, records):
            hidden = self.encoder.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
            per_record = self.head(
                hidden, input_ids, attention_mask, records, self.encoder.get_output_embeddings().weight.detach()
            )
            flat = [question for record in per_record for question in record]
            padded = torch.full((len(flat), max(len(z) for z in flat)), -1e4)
            for i, z in enumerate(flat):
                padded[i, : len(z)] = z
            return padded


    def build_toy_model(vocab_size, seed=0):
        torch.manual_seed(seed)
        backbone = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=vocab_size,
                hidden_size=TOY_HIDDEN,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                max_position_embeddings=1024,
                tie_word_embeddings=True,
            )
        )
        # decision_from_lm.py처럼 헤드 생성은 별도 난수 상태에서 random_state로 고정합니다.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3407)
            head = clef_src.JointSchemaHead(**from_lm_src.default_head_config(TOY_HIDDEN, width=128))
        return ToyClefDecisionModel(backbone, head.float())
    return ToyClefDecisionModel, build_toy_model


@app.cell
def _(build_toy_model, mo, tokenizer):
    preview_model = build_toy_model(len(tokenizer)).eval()
    _backbone = sum(_p.numel() for _p in preview_model.encoder.parameters())
    _head = sum(_p.numel() for _p in preview_model.head.parameters())
    mo.md(
        f"백본 {_backbone / 1e6:.1f}M (대부분 248,077 × 128 임베딩), 헤드 {_head / 1e6:.2f}M 매개변수. "
        f"헤드 설정: `{preview_model.head.layers[0].self_attn.num_heads}` heads, "
        f"routing 층 {len(preview_model.head.evidence_layers)}개, decoder 층 {len(preview_model.head.layers)}개"
    )
    return (preview_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 헤드 순전파를 한 단계씩 따라가기

    아래 `trace_head`는 원본 `JointSchemaHead.forward`(`joint_schema_model.py`)를 레코드 하나에 대해
    그대로 풀어 쓴 것입니다. 단계별 텐서 shape를 기록하고, 마지막에 원본 헤드 출력과 같은지 확인합니다.
    표의 `L`은 토큰 수, `H`는 백본 은닉 크기(128), `W`는 헤드 폭(128), `Q`는 질문 수, `P`는 전체 선택지 수입니다.

    핵심 아이디어는 세 가지입니다.

    - **증거 라우팅(evidence routing)**: 각 선택지 벡터가 전체 토큰 기억(memory)에 cross-attention해
      state 속에서 자신을 뒷받침하는 근거를 모읍니다.
    - **공동 판단(joint decision)**: 질문 벡터(field)들이 `TransformerDecoderLayer`에서 **서로 self-attention**
      하고 memory에 cross-attention합니다. 그래서 "환불 요청이면 billing 팀"처럼 질문 사이의 관계를 반영할 수 있습니다.
    - **점수 = 사전 점수 + 게이트 × (결합 점수)**: 사전 점수(prior)는 선택지 단어 임베딩과 질문의 코사인 유사도,
      결합 점수는 field와 선택지 벡터의 코사인 유사도와 작은 MLP(residual scorer) 점수입니다.
      초기에는 `prior_logit_scale = joint_logit_scale = 0`(배율 1), `residual_gate = 0`(시그모이드 0.5)입니다.
    """)
    return


@app.cell
def _(F, clef_src, math, mo, preview_model, preview_record, tokenizer, torch):
    @torch.no_grad()
    def trace_head(head, hidden_states, input_ids, record, output_embedding_weight):
        """JointSchemaHead.forward를 레코드 하나에 대해 풀어 쓰고 shape를 기록합니다."""
        steps = []

        def note(name, tensor, symbol, meaning):
            steps.append({"단계": name, "shape": str(tuple(tensor.shape)), "기호": symbol, "의미": meaning})

        h = head.hidden_norm(hidden_states[0, : len(record.input_ids)])
        note("hidden_norm(hidden)", h, "[L, H]", "백본 마지막 은닉 상태를 LayerNorm")
        memory = head.memory_projection(h).unsqueeze(0)
        note("memory_projection", memory, "[1, L, W]", "모든 토큰의 기억. 이후 cross-attention의 key/value")
        global_vector = h[-1]
        note("global_vector", global_vector, "[H]", "마지막 토큰(JOINT SCHEMA DECISIONS:)의 상태")
        question_vectors = torch.stack([h[slice(*q.question_span)].mean(0) for q in record.questions])
        note("question_vectors", question_vectors, "[Q, H]", "질문 문장 span 평균")
        spans = [span for q in record.questions for span in q.option_spans]
        counts = [len(q.option_spans) for q in record.questions]
        contexts = torch.stack([h[a:b].mean(0) for a, b in spans])
        note("option contexts", contexts, "[P, H]", "선택지 span의 은닉 상태 평균 (문맥 속 의미)")
        lexical = torch.stack([output_embedding_weight[input_ids[0, a:b]].mean(0) for a, b in spans])
        note("option lexical", lexical, "[P, H]", "선택지 토큰의 출력 임베딩 평균 (단어 자체의 의미)")
        owner = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        routed = (
            head.option_context_projection(contexts)
            + head.option_lexical_projection(lexical)
            + head.option_question_projection(question_vectors)[owner]
        ).unsqueeze(0)
        note("option queries", routed, "[1, P, W]", "문맥 + 단어 + 소속 질문을 더한 선택지 질의")
        for layer in head.evidence_layers:
            routed = layer(routed, memory)
        note("evidence_layers ×2", routed, "[1, P, W]", "선택지가 memory에 cross-attention해 근거 수집")
        split_options = torch.split(routed[0], counts)
        base_fields = head.question_projection(question_vectors)
        summaries = []
        for field, options in zip(base_fields, split_options):
            weights = torch.softmax(options @ field / math.sqrt(options.shape[-1]), dim=0)
            summaries.append((weights.unsqueeze(-1) * options).sum(0))
        type_ids = torch.tensor([q.question_type for q in record.questions])
        fields = (
            base_fields
            + head.option_summary_norm(torch.stack(summaries))
            + head.global_projection(global_vector).unsqueeze(0)
            + head.type_embedding(type_ids)
        ).unsqueeze(0)
        note("fields", fields, "[1, Q, W]", "질문 + 선택지 요약 + 전체 요약 + 유형 임베딩")
        for layer in head.layers:
            fields = layer(fields, memory)
        fields = head.field_norm(fields[0])
        note("decoder layers ×4", fields, "[Q, W]", "질문끼리 self-attention(공동 판단) + memory cross-attention")
        prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        gate = torch.sigmoid(head.residual_gate)
        logits = []
        for index, (field, options, lexical_options) in enumerate(
            zip(fields, split_options, torch.split(lexical, counts))
        ):
            anchor = F.normalize(question_vectors[index] + global_vector, dim=-1)
            prior = prior_scale * (F.normalize(lexical_options, dim=-1) @ anchor)
            options = head.option_norm(options)
            repeated = field.unsqueeze(0).expand_as(options)
            cosine = F.cosine_similarity(repeated, options, dim=-1)
            residual = head.residual_scorer(
                torch.cat([repeated, options, repeated * options, (repeated - options).abs()], dim=-1)
            ).squeeze(-1)
            logits.append(prior + gate * (joint_scale * cosine + residual))
            note(f"logits[{record.questions[index].question_id}]", logits[-1], "[선택지 수]", "prior + gate·(scale·cos + residual)")
        return steps, logits


    mo.stop(preview_record is None, mo.callout("4부의 인코딩이 실패해 건너뜁니다.", kind="warn"))
    _batch = clef_src.collate_records([preview_record], tokenizer.pad_token_id, torch.device("cpu"))
    with torch.no_grad():
        _hidden = preview_model.encoder.model(
            input_ids=_batch["input_ids"], attention_mask=_batch["attention_mask"], use_cache=False
        ).last_hidden_state
        _embedding = preview_model.encoder.get_output_embeddings().weight
        _reference = preview_model.head(_hidden, _batch["input_ids"], _batch["attention_mask"], _batch["records"], _embedding)[0]
    _steps, _manual = trace_head(preview_model.head, _hidden, _batch["input_ids"], preview_record, _embedding)
    _same = all(torch.allclose(_a, _b, atol=1e-5) for _a, _b in zip(_manual, _reference))
    mo.vstack(
        [
            mo.ui.table(_steps, selection=None, page_size=20),
            mo.md(f"원본 `JointSchemaHead.forward`와 결과 일치: **{_same}**"),
            mo.md(
                "학습 전 확률: "
                + ", ".join(
                    f"`{_q.question_id}` {dict(zip(_q.option_ids, [round(_p, 3) for _p in _z.softmax(-1).tolist()]))}"
                    for _q, _z in zip(preview_record.questions, _reference)
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    학습 전 확률은 거의 균등합니다. 헤드가 무작위로 초기화되어 있기 때문입니다. Unsloth의 실제 헤드
    (`clef.py`)는 같은 계산을 배치 전체에 대해 한 번에 수행하고, 긴 입력에서 메모리를 아끼도록 LayerNorm과
    span 평균을 토큰 묶음(chunk) 단위로 계산하는 사용자 정의 연산을 씁니다. 가중치 구조(state dict)는 원본과 같습니다.

    ## 6. 손실 함수: `_decision_loss`

    `DecisionTrainer.compute_loss`는 선택지 logit과 목표 분포로 손실을 계산합니다. 기본은 soft target에 대한
    교차 엔트로피이고, Cloudflare Clef 레시피의 세 가지 항을 선택적으로 더할 수 있습니다.

    | 인자 | 수식 | 효과 |
    | --- | --- | --- |
    | `label_smoothing` = ε | target ← (1−ε)·target + ε·균등분포 | 과도한 확신 억제 |
    | `brier_weight` | + w·Σ(p − target)² | 확률 자체를 정답에 맞춤 → 보정 개선 |
    | `ordinal_weight` | + w·E[|예측 등급 − 정답 등급|] / (등급 수 − 1) | `score` 질문에서 먼 등급 오답에 더 큰 벌점 |

    Unsloth 2026.10.9의 기본값(`CLEF_RECIPE = {}`)은 세 항 모두 0입니다. 아래 함수는 `decision.py`의
    `_decision_loss`를 그대로 옮긴 것입니다. 정답이 등급 2인 `score` 질문에서 예측 분포별 손실을 비교해 보세요.
    """)
    return


@app.cell
def _(torch):
    # unsloth/models/decision.py의 _decision_loss (AGPL-3.0, Copyright Daniel Han-Chen & the Unsloth team)
    def decision_loss(logits, target, mask, ordinal=None, label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0):
        log_p = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
        smoothed = target
        if label_smoothing:
            uniform = mask.float() / mask.sum(-1, keepdim=True).clamp(min=1)
            smoothed = (1.0 - label_smoothing) * target + label_smoothing * uniform
        loss = -(smoothed * log_p).sum(-1).mean()
        p = log_p.exp()
        if brier_weight:
            loss = loss + brier_weight * ((p - target) ** 2 * mask).sum(-1).mean()
        if ordinal_weight and ordinal is not None and ordinal.any():
            levels = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
            distance = (levels[:, None] - levels[None, :]).abs()
            span = (mask.sum(-1) - 1).clamp(min=1).to(p.dtype)
            expected = torch.einsum("ri,ij,rj->r", p, distance, target) / span
            loss = loss + ordinal_weight * (expected * ordinal).sum() / ordinal.sum()
        return loss
    return (decision_loss,)


@app.cell
def _(mo):
    loss_smoothing = mo.ui.slider(0.0, 0.3, step=0.05, value=0.0, label="label_smoothing", show_value=True)
    loss_brier = mo.ui.slider(0.0, 1.0, step=0.1, value=0.0, label="brier_weight", show_value=True)
    loss_ordinal = mo.ui.slider(0.0, 1.0, step=0.1, value=0.5, label="ordinal_weight", show_value=True)
    mo.hstack([loss_smoothing, loss_brier, loss_ordinal])
    return loss_brier, loss_ordinal, loss_smoothing


@app.cell
def _(decision_loss, loss_brier, loss_ordinal, loss_smoothing, mo, torch):
    _target = torch.tensor([[0.0, 0.0, 1.0]])
    _mask = torch.ones(1, 3, dtype=torch.bool)
    _ordinal = torch.tensor([True])
    _cases = {
        "정답 확신 [.05, .05, .90]": [0.05, 0.05, 0.90],
        "균등 [1/3, 1/3, 1/3]": [1 / 3, 1 / 3, 1 / 3],
        "인접 오답 [.05, .90, .05]": [0.05, 0.90, 0.05],
        "먼 오답 [.90, .05, .05]": [0.90, 0.05, 0.05],
    }
    _rows = []
    for _name, _p in _cases.items():
        _logits = torch.tensor([_p]).log()
        _rows.append(
            {
                "예측 분포": _name,
                "교차 엔트로피만": round(float(decision_loss(_logits, _target, _mask)), 3),
                "선택한 설정": round(
                    float(
                        decision_loss(
                            _logits, _target, _mask, _ordinal, loss_smoothing.value, loss_brier.value, loss_ordinal.value
                        )
                    ),
                    3,
                ),
            }
        )
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md("교차 엔트로피만 쓰면 인접 오답과 먼 오답의 손실이 같습니다. `ordinal_weight`를 올리면 먼 오답의 손실이 더 커집니다."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. CPU 미니 학습: 헤드 먼저, 그다음 백본

    `decision_from_lm.py`의 `freeze_backbone`은 "from-LM 레시피의 1단계"로, 새 헤드를 **고정된 백본 특징**
    위에서 먼저 학습시킵니다. 고정 전 학습 가능했던 매개변수 이름을 기억해 두었다가 `unfreeze_backbone`이
    정확히 그것들만 다시 학습 가능하게 합니다(LoRA를 붙인 경우 LoRA 가중치만 다시 풀립니다).

    여기서는 고객 문의 문장을 템플릿으로 만들어 세 질문(`team`, `refund`, `urgency`)을 학습합니다.
    옵티마이저는 `DecisionTrainer.create_optimizer`처럼 **백본(encoder.*)과 헤드에 서로 다른 학습률**을 씁니다.

    - **1단계**: `freeze_backbone` → 헤드만 학습
    - **2단계**: `unfreeze_backbone` → 백본과 헤드를 함께 학습

    이 실습의 백본은 사전 학습되지 않은 무작위 모델이라 1단계의 효과가 제한적입니다. 실제 LLM은
    은닉 상태에 이미 의미가 담겨 있어 헤드만 학습해도 효과가 큽니다. 아래 데이터 생성 코드를 먼저 확인하세요.
    """)
    return


@app.cell
def _(TOY_QUESTIONS, mo, random):
    TOPIC_TEMPLATES = {
        "billing": [
            "I was charged twice for invoice #{n}.",
            "My card payment for order {n} failed but money left my account.",
            "Invoice {n} shows the wrong amount.",
        ],
        "technical": [
            "The app crashes when I open the dashboard.",
            "I get error {n} every time I log in.",
            "Your API has been down since this morning.",
        ],
        "sales": [
            "What does the enterprise plan cost for {n} seats?",
            "Can I get a discount if I upgrade to the pro plan?",
            "Do you offer annual pricing for teams?",
        ],
    }
    REFUND_TEMPLATES = {True: ["Please refund the money.", "I want my money back."], False: ["", "Just tell me what to do."]}
    URGENCY_TEMPLATES = {
        0: ["No rush, whenever you have time.", "It can wait."],
        1: ["Please look at this this week.", "I'd like an answer soon."],
        2: ["I need this fixed today!", "This is urgent, today please."],
    }
    # augment_row가 사용할 바꿔 쓰기·대체 ID (decision_datasets.py의 _row와 같은 형식)
    TOY_VARIANTS = {
        "team": {"instructions": ["Who should take this ticket?", "Route this message to a team."], "ids": ["queue", "department"]},
        "refund": {"instructions": ["Is the customer asking for money back?"], "ids": ["wants_refund", "money_back"]},
        "urgency": {"instructions": ["How soon must we reply?"], "ids": ["priority", "deadline"]},
    }


    def make_ticket(rng):
        team = rng.choice(list(TOPIC_TEMPLATES))
        refund = rng.random() < (0.5 if team == "billing" else 0.15)
        urgency = rng.randrange(3)
        parts = [
            rng.choice(TOPIC_TEMPLATES[team]).format(n=rng.randint(100, 9999)),
            rng.choice(REFUND_TEMPLATES[refund]),
            rng.choice(URGENCY_TEMPLATES[urgency]),
        ]
        return {
            "state": " ".join(part for part in parts if part),
            "questions": TOY_QUESTIONS,
            "gold": {"team": team, "refund": refund, "urgency": urgency},
            "_variants": TOY_VARIANTS,
        }


    _rng = random.Random(0)
    toy_splits = {
        "train": [make_ticket(_rng) for _ in range(400)],
        "calibration": [make_ticket(_rng) for _ in range(80)],
        "test": [make_ticket(_rng) for _ in range(80)],
    }
    mo.ui.table(
        [{"state": _r["state"], **{_k: str(_v) for _k, _v in _r["gold"].items()}} for _r in toy_splits["train"][:6]],
        selection=None,
    )
    return (toy_splits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    다음 셀은 학습에 쓰는 함수들입니다. `encode_rows`는 `FastDecisionModel.build_dataset`의 Clef 경로
    (`_clef_items`)를 줄인 것이고, `collate`는 `ClefDataCollator`처럼 레코드 여러 개를 한 배치로 묶으면서
    **모든 레코드의 모든 질문**을 `[전체 질문 수, 최대 선택지 수]` 목표 텐서로 펼칩니다. `evaluate_rows`는
    평가 결과를 Unsloth의 지표 함수(`_metrics`, `_calibrate_clef`)가 받는 형식으로 돌려줍니다.
    """)
    return


@app.cell
def _(clef_src, common_src, datasets_src, decision_loss, mo, random, time, tokenizer, torch):
    TOY_MAX_LEN = 768  # 증강의 파생 질문까지 넣어도 스키마가 들어가는 길이
    # Clef 질문 유형 이름 → Unsloth 지표 함수의 유형 번호(choice=0, score=1, noul=2)
    METRIC_QTYPE = {_kind: common_src.QUESTION_TYPES.index(_kind) for _kind in common_src.QUESTION_TYPES}


    def encode_rows(rows, augment_rng=None):
        items = []
        for index, row in enumerate(rows):
            if augment_rng is not None:
                row = datasets_src.augment_row(row, augment_rng)
            questions = {name: common_src._clef_question(q) for name, q in row["questions"].items()}
            record = clef_src.encode_record(tokenizer, {"state": row["state"], "questions": questions}, max_length=TOY_MAX_LEN)
            targets, labels, kinds = [], [], []
            for question in record.questions:
                kind = questions[question.question_id]["type"]
                target, label = common_src._target_for(kind, list(question.option_ids), row["gold"][question.question_id])
                targets.append(target)
                labels.append(label)
                kinds.append(kind)
            items.append({"record": record, "targets": targets, "labels": labels, "kinds": kinds, "row": index})
        return items


    def collate(items):
        batch = clef_src.collate_records([item["record"] for item in items], tokenizer.pad_token_id, torch.device("cpu"))
        targets = [target for item in items for target in item["targets"]]
        width = max(len(target) for target in targets)
        batch["target"] = torch.zeros(len(targets), width)
        batch["marker_mask"] = torch.zeros(len(targets), width, dtype=torch.bool)
        for i, target in enumerate(targets):
            batch["target"][i, : len(target)] = torch.tensor(target)
            batch["marker_mask"][i, : len(target)] = True
        batch["ordinal"] = torch.tensor([kind == "score" for item in items for kind in item["kinds"]])
        return batch


    @torch.no_grad()
    def evaluate_rows(model, items, batch_size=16):
        """질문별 logit과, _metrics가 받는 {target, label, qtype, row} 목록을 돌려줍니다."""
        was_training = model.training
        model.eval()
        logits, questions = [], []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = collate(chunk)
            z = model(batch["input_ids"], batch["attention_mask"], batch["records"])
            k = 0
            for item in chunk:
                for target, label, kind in zip(item["targets"], item["labels"], item["kinds"]):
                    logits.append(z[k, : len(target)])
                    questions.append(
                        {"target": target, "label": label, "qtype": METRIC_QTYPE[kind], "row": item["row"], "kind": kind}
                    )
                    k += 1
        model.train(was_training)
        return logits, questions


    def accuracy_by_type(logits, questions):
        result = {}
        for kind in ("choice", "noul", "score"):
            hits = [int(z.argmax()) == q["label"] for z, q in zip(logits, questions) if q["kind"] == kind]
            result[kind] = round(sum(hits) / max(1, len(hits)), 3)
        records = {}
        for z, q in zip(logits, questions):
            records[q["row"]] = records.get(q["row"], True) and int(z.argmax()) == q["label"]
        result["record (모든 질문 정답)"] = round(sum(records.values()) / max(1, len(records)), 3)
        return result


    def make_optimizer(model, backbone_lr, head_lr):
        # DecisionTrainer.create_optimizer처럼 encoder.* 와 헤드의 학습률을 나눕니다.
        groups = {True: [], False: []}
        for name, param in model.named_parameters():
            if param.requires_grad:
                groups[name.startswith("encoder.")].append(param)
        return torch.optim.AdamW(
            [
                {"params": params, "lr": backbone_lr if is_encoder else head_lr}
                for is_encoder, params in groups.items()
                if params
            ],
            weight_decay=0.01,
        )


    def train_stage(model, train_rows, steps, backbone_lr, head_lr, loss_kwargs, seed, augment, title, batch_size=8):
        """학습 가능한 매개변수만 갱신하는 짧은 학습 루프. 손실 기록과 시간을 돌려줍니다."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if steps <= 0:
            return {"losses": [], "seconds": 0.0, "trainable": trainable}
        optimizer = make_optimizer(model, backbone_lr, head_lr)
        rng = random.Random(seed)
        fixed_items = None if augment else encode_rows(train_rows)
        losses, started = [], time.time()
        model.train()
        for _step in mo.status.progress_bar(range(steps), title=title, remove_on_exit=True):
            if augment:  # 증강은 배치마다 새로 적용합니다.
                items = encode_rows(rng.sample(train_rows, batch_size), augment_rng=rng)
            else:
                items = rng.sample(fixed_items, batch_size)
            batch = collate(items)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["records"])
            loss = decision_loss(logits, batch["target"], batch["marker_mask"], batch["ordinal"], **loss_kwargs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        return {"losses": losses, "seconds": round(time.time() - started, 1), "trainable": trainable}
    return (
        TOY_MAX_LEN,
        accuracy_by_type,
        collate,
        encode_rows,
        evaluate_rows,
        train_stage,
    )


@app.cell
def _(mo):
    def loss_svg(stages, width=640, height=220):
        """단계별 손실 목록을 이어 그린 간단한 SVG 선 그래프."""
        colors = ["#3b82f6", "#f97316", "#10b981"]
        values = [v for losses in stages.values() for v in losses]
        if not values:
            return mo.md("기록된 손실이 없습니다.")
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        pad = 30
        step_x = (width - 2 * pad) / max(1, len(values) - 1)
        parts, legend, offset = [], [], 0
        for index, (name, losses) in enumerate(stages.items()):
            if not losses:
                continue
            points = " ".join(
                f"{pad + (offset + i) * step_x:.1f},{height - pad - (v - low) / span * (height - 2 * pad):.1f}"
                for i, v in enumerate(losses)
            )
            color = colors[index % len(colors)]
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>')
            legend.append(f'<tspan fill="{color}">■ {name}</tspan> ')
            offset += len(losses)
        axis = (
            f'<text x="{pad}" y="16" font-size="12" fill="currentColor">{"".join(legend)}</text>'
            f'<text x="2" y="{pad + 4}" font-size="11" fill="currentColor">{high:.2f}</text>'
            f'<text x="2" y="{height - pad + 4}" font-size="11" fill="currentColor">{low:.2f}</text>'
            f'<text x="{width - pad - 60}" y="{height - 8}" font-size="11" fill="currentColor">스텝 →</text>'
        )
        return mo.Html(
            f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px" role="img">{axis}{"".join(parts)}</svg>'
        )
    return (loss_svg,)


@app.cell
def _(mo):
    toy_stage_a_steps = mo.ui.slider(0, 150, step=10, value=60, label="1단계(헤드만) 스텝", show_value=True)
    toy_stage_b_steps = mo.ui.slider(0, 300, step=20, value=120, label="2단계(백본+헤드) 스텝", show_value=True)
    toy_backbone_lr = mo.ui.dropdown({"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3}, value="1e-3", label="백본 학습률")
    toy_head_lr = mo.ui.dropdown({"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3}, value="1e-3", label="헤드 학습률")
    toy_augment = mo.ui.checkbox(label="스키마 증강(augment_row) 사용")
    toy_use_loss_settings = mo.ui.checkbox(label="6부 슬라이더의 손실 설정 사용")
    run_toy_training = mo.ui.run_button(label="미니 학습 실행")
    mo.vstack(
        [
            mo.hstack([toy_stage_a_steps, toy_stage_b_steps]),
            mo.hstack([toy_backbone_lr, toy_head_lr]),
            mo.hstack([toy_augment, toy_use_loss_settings]),
            run_toy_training,
        ]
    )
    return (
        run_toy_training,
        toy_augment,
        toy_backbone_lr,
        toy_head_lr,
        toy_stage_a_steps,
        toy_stage_b_steps,
        toy_use_loss_settings,
    )


@app.cell
def _(
    accuracy_by_type,
    build_toy_model,
    encode_rows,
    evaluate_rows,
    from_lm_src,
    json,
    loss_brier,
    loss_ordinal,
    loss_smoothing,
    loss_svg,
    mo,
    new_run_dir,
    run_toy_training,
    tokenizer,
    toy_augment,
    toy_backbone_lr,
    toy_head_lr,
    toy_splits,
    toy_stage_a_steps,
    toy_stage_b_steps,
    toy_use_loss_settings,
    train_stage,
):
    mo.stop(not run_toy_training.value, mo.md("**미니 학습 실행** 버튼을 누르면 CPU에서 학습을 시작합니다(기본 설정 약 1분)."))
    _loss_kwargs = (
        {"label_smoothing": loss_smoothing.value, "brier_weight": loss_brier.value, "ordinal_weight": loss_ordinal.value}
        if toy_use_loss_settings.value
        else {}
    )
    toy_model = build_toy_model(len(tokenizer))
    _test_items = encode_rows(toy_splits["test"])
    _common = {"loss_kwargs": _loss_kwargs, "augment": toy_augment.value}
    toy_history = {"학습 전": accuracy_by_type(*evaluate_rows(toy_model, _test_items))}

    # 1단계: decision_from_lm.freeze_backbone — 백본을 고정하고 헤드만 학습
    from_lm_src.freeze_backbone(toy_model)
    _stage_a = train_stage(
        toy_model, toy_splits["train"], toy_stage_a_steps.value, toy_backbone_lr.value, toy_head_lr.value,
        seed=1, title="1단계: 헤드만", **_common,
    )
    toy_history["1단계 후"] = accuracy_by_type(*evaluate_rows(toy_model, _test_items))

    # 2단계: unfreeze_backbone — freeze 전에 학습 가능했던 매개변수만 다시 풀기
    from_lm_src.unfreeze_backbone(toy_model)
    _stage_b = train_stage(
        toy_model, toy_splits["train"], toy_stage_b_steps.value, toy_backbone_lr.value, toy_head_lr.value,
        seed=2, title="2단계: 백본 + 헤드", **_common,
    )
    toy_history["2단계 후"] = accuracy_by_type(*evaluate_rows(toy_model, _test_items))

    _run_dir = new_run_dir("decision-internals-cpu")
    _summary = {
        "settings": {
            "stage_a_steps": toy_stage_a_steps.value,
            "stage_b_steps": toy_stage_b_steps.value,
            "backbone_lr": toy_backbone_lr.value,
            "head_lr": toy_head_lr.value,
            "augment": toy_augment.value,
            "loss_kwargs": _loss_kwargs,
        },
        "accuracy": toy_history,
        "stage_a": _stage_a,
        "stage_b": _stage_b,
    }
    (_run_dir / "cpu_mini_training.json").write_text(json.dumps(_summary, indent=2, ensure_ascii=False))
    mo.vstack(
        [
            mo.md(
                f"1단계 학습 매개변수 **{_stage_a['trainable'] / 1e6:.2f}M** ({_stage_a['seconds']}초), "
                f"2단계 학습 매개변수 **{_stage_b['trainable'] / 1e6:.2f}M** ({_stage_b['seconds']}초). "
                f"결과: `{_run_dir}`"
            ),
            mo.ui.table([{"시점": _k, **_v} for _k, _v in toy_history.items()], selection=None),
            loss_svg({"1단계(헤드만)": _stage_a["losses"], "2단계(백본+헤드)": _stage_b["losses"]}),
        ]
    )
    return toy_history, toy_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    생각해 볼 점:

    - 1단계에서 학습 매개변수 수가 헤드 크기와 같은지, 2단계에서 백본이 다시 포함되는지 확인하세요.
    - 1단계 스텝을 0으로 두고 2단계만 학습하면 결과가 어떻게 달라지나요?
    - 스키마 증강을 켜면 같은 스텝에서 정확도가 낮아질 수 있습니다. 선택지 이름이 `A, B, C`로 바뀌는 등
      문제가 어려워지기 때문입니다. 대신 처음 보는 스키마에 대한 일반화가 좋아집니다.
    - `record` 정확도는 한 행의 모든 질문을 맞힌 비율입니다. 질문별 정확도보다 항상 낮거나 같습니다.

    ## 8. 온도 보정과 추론

    분류 정확도가 높아도 확률은 과신(overconfident)하거나 과소 확신할 수 있습니다. **보정(calibration)** 은
    "신뢰도 0.9로 낸 답이 실제로 약 90% 맞도록" 확률을 조정하는 것입니다. 측정 지표 **ECE**는 신뢰도를
    15개 구간으로 나눠 구간별 |평균 신뢰도 − 정확도|를 표본 비율로 가중 평균한 값이며, 낮을수록 좋습니다.

    Unsloth의 `FastDecisionModel.calibrate`(Clef 경로는 `_calibrate_clef`)는 **온도 스케일링**을 씁니다.
    logit을 온도 T로 나눈 뒤 softmax합니다(T > 1이면 확률이 완만해지고, T < 1이면 날카로워집니다).

    1. 학습에 쓰지 않은 홀드아웃에서 정답 라벨(hard label)에 대한 음의 로그 우도를 최소화하는 **헤드 전체 온도**를
       LBFGS로 찾습니다(0.05~20으로 제한).
    2. 그 위에 질문 유형별 **상대 온도**를 다시 맞춥니다(유형별 10개 이상일 때만, 0.5~5로 제한).
    3. 보고하는 지표는 홀드아웃을 반으로 나눠 **한쪽에서 맞춘 온도로 다른 쪽을 평가**한 값입니다.

    아래는 7부에서 학습한 모델에 실제 `_calibrate_clef`를 적용하고, 별도 테스트 세트에서 보정 전후를 비교합니다.
    """)
    return


@app.cell
def _(common_src, encode_rows, evaluate_rows, mo, toy_model, toy_splits):
    toy_decision_config = {"temperature": [1.0, 1.0, 1.0]}  # load_lm_as_decision_model의 초기값
    _cal_logits, _cal_items = evaluate_rows(toy_model, encode_rows(toy_splits["calibration"]))
    _report = common_src._calibrate_clef(toy_decision_config, _cal_logits, _cal_items)
    _test_logits, _test_items = evaluate_rows(toy_model, encode_rows(toy_splits["test"]))
    _before = common_src._metrics(_test_logits, _test_items, [1.0] * len(_test_logits))
    _served = common_src._served_temperatures(toy_decision_config, _test_logits, _test_items)
    _after = common_src._metrics(_test_logits, _test_items, _served)


    def _reliability(temperatures, bins=5):
        rows = [{"n": 0, "conf": 0.0, "hit": 0} for _ in range(bins)]
        for z, item, t in zip(_test_logits, _test_items, temperatures):
            p = (z / t).softmax(-1)
            confidence = float(p.max())
            row = rows[min(bins - 1, int(confidence * bins))]
            row["n"] += 1
            row["conf"] += confidence
            row["hit"] += int(int(p.argmax()) == item["label"])
        return [
            f"{r['conf'] / r['n']:.2f} / {r['hit'] / r['n']:.2f} ({r['n']})" if r["n"] else "-" for r in rows
        ]


    _bins_before, _bins_after = _reliability([1.0] * len(_test_logits)), _reliability(_served)
    mo.vstack(
        [
            mo.md(
                f"맞춘 온도: 헤드 전체 `{toy_decision_config['head_temperature']:.3f}`, "
                f"유형별 상대 온도(choice, score, noul) `{[round(_t, 3) for _t in toy_decision_config['temperature']]}`"
            ),
            mo.ui.table(
                [
                    {"테스트 세트": "보정 전 (T=1)", **{_k: round(_v, 4) for _k, _v in _before.items()}},
                    {"테스트 세트": "보정 후", **{_k: round(_v, 4) for _k, _v in _after.items()}},
                ],
                selection=None,
            ),
            mo.md("신뢰도 구간별 **평균 신뢰도 / 실제 정확도 (개수)** — 두 값이 가까울수록 잘 보정된 것입니다."),
            mo.ui.table(
                [
                    {"": "보정 전", **{f"{_i / 5:.1f}~{(_i + 1) / 5:.1f}": _v for _i, _v in enumerate(_bins_before)}},
                    {"": "보정 후", **{f"{_i / 5:.1f}~{(_i + 1) / 5:.1f}": _v for _i, _v in enumerate(_bins_after)}},
                ],
                selection=None,
            ),
        ]
    )
    return (toy_decision_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    정확도는 온도와 무관합니다. 온도로 나눠도 logit의 순위가 바뀌지 않기 때문입니다. 보정은 확률의 크기만 바꿉니다.
    테스트 세트가 작아(80행) ECE가 실행마다 흔들릴 수 있습니다.

    ### 추론: `FastDecisionModel.predict`의 흐름

    `predict`(Clef 경로는 `_clef_decide`)는 다음 순서로 동작합니다. 아래 `toy_predict`는 이를 그대로 옮긴 것입니다.

    1. `encode_record`로 모든 질문을 한 프롬프트로 인코딩 → 2. 한 번의 순전파 →
    3. 질문마다 보정 온도로 나눠 softmax → 4. `systemone_answer`로 응답 형식 변환 → 5. `answer`, `probabilities` 추가

    학습 템플릿에 없는 문장도 넣어 보세요. 이 장난감 모델은 템플릿의 단어 몇 개에 의존하므로 쉽게 틀립니다.
    """)
    return


@app.cell
def _(mo):
    toy_predict_state = mo.ui.text_area(
        value="My invoice 4411 was charged twice. I want my money back today!", label="state", full_width=True
    )
    toy_predict_state
    return (toy_predict_state,)


@app.cell
def _(
    TOY_MAX_LEN,
    TOY_QUESTIONS,
    clef_src,
    common_src,
    mo,
    tokenizer,
    toy_decision_config,
    toy_model,
    toy_predict_state,
    torch,
):
    @torch.no_grad()
    def toy_predict(model, state, questions, config):
        """decision.py의 _clef_decide + _predicted를 CPU 실습용으로 옮긴 것."""
        questions = {str(name): common_src._clef_question(q) for name, q in questions.items()}
        record = clef_src.encode_record(tokenizer, {"state": state, "questions": questions}, max_length=TOY_MAX_LEN)
        batch = clef_src.collate_records([record], tokenizer.pad_token_id, torch.device("cpu"))
        model.eval()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["records"])
        rows = [logits[i, : len(q.option_ids)] for i, q in enumerate(record.questions)]
        kinds = [{"qtype": common_src.QUESTION_TYPES.index(questions[q.question_id]["type"])} for q in record.questions]
        answers = {}
        for question, row, scale in zip(record.questions, rows, common_src._served_temperatures(config, rows, kinds)):
            probabilities = dict(zip(question.option_ids, (row / scale).softmax(-1).tolist()))
            asked = questions[question.question_id]
            answers[question.question_id] = common_src._predicted(
                asked, clef_src.systemone_answer(asked, probabilities), probabilities
            )
        return answers


    _answers = toy_predict(toy_model, toy_predict_state.value, TOY_QUESTIONS, toy_decision_config)
    mo.ui.table(
        [
            {"질문": _name, "answer": str(_a["answer"]), "probabilities": {_k: round(_v, 3) for _k, _v in _a["probabilities"].items()}}
            for _name, _a in _answers.items()
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. GPU 실습: Unsloth API로 같은 과정 실행하기

    지금까지 손으로 따라간 과정을 실제 Unsloth API로 실행합니다. 각 단계가 앞에서 본 어느 부분에 해당하는지 표시했습니다.

    | 단계 | API | 앞 실습 |
    | --- | --- | --- |
    | 모델 생성 | `FastDecisionModel.from_pretrained` → `load_lm_as_decision_model` | 5부 |
    | LoRA | `FastDecisionModel.get_peft_model` | – |
    | 데이터 | `load_source` / `build_decision_mixture` → `build_dataset` → `split_holdout` | 2~4부 |
    | 1단계 학습 | `FastDecisionModel.freeze_backbone` + `DecisionTrainer` | 6~7부 |
    | 2단계 학습 | `FastDecisionModel.unfreeze_backbone` + `DecisionTrainer` | 6~7부 |
    | 평가·보정·추론 | `evaluate`, `calibrate`, `predict` | 8부 |

    - 4비트 LoRA에서 1단계 `freeze_backbone`은 LoRA 가중치를 고정하고, `unfreeze_backbone`은 LoRA 가중치만 다시 풉니다.
      4비트 원본 가중치는 처음부터 학습되지 않습니다.
    - `DecisionTrainer`는 `head_learning_rate`(기본 1e-4)를 헤드에, `learning_rate`를 LoRA에 적용합니다.
    - 기본 백본은 12GB GPU에서도 빠르게 돌도록 `unsloth/Qwen3.5-0.8B`입니다. 원본 노트북과 같은 4B는 목록에서 고르세요.
      설정을 바꾼 뒤에는 커널을 재시작하고 다시 실행하세요.
    - GPU가 있으면 노트북 첫 셀이 unsloth를 transformers보다 먼저 import합니다.
    """)
    return


@app.cell
def _(mo):
    gpu_model_name = mo.ui.dropdown(
        ["unsloth/Qwen3.5-0.8B", "unsloth/Qwen3.5-2B", "unsloth/Qwen3.5-4B"], value="unsloth/Qwen3.5-0.8B", label="백본"
    )
    gpu_train_rows = mo.ui.number(start=0, stop=100000, step=500, value=0, label="학습 행 수 (0 = 전체)")
    gpu_use_mixture = mo.ui.checkbox(label="공개 데이터 혼합 (typed_decisions 80% + BoolQ 10% + SST-5 10%, 증강 포함)")
    gpu_stage_a_steps = mo.ui.number(start=0, stop=500, value=20, label="1단계 스텝 (헤드만)")
    gpu_stage_b_steps = mo.ui.number(start=1, stop=2000, value=60, label="2단계 스텝 (LoRA + 헤드)")
    gpu_batch_size = mo.ui.number(start=1, stop=32, value=8, label="per_device_train_batch_size")
    run_gpu = mo.ui.run_button(label="GPU 실습 실행")
    mo.vstack(
        [
            mo.hstack([gpu_model_name, gpu_train_rows]),
            gpu_use_mixture,
            mo.hstack([gpu_stage_a_steps, gpu_stage_b_steps, gpu_batch_size]),
            run_gpu,
        ]
    )
    return (
        gpu_batch_size,
        gpu_model_name,
        gpu_stage_a_steps,
        gpu_stage_b_steps,
        gpu_train_rows,
        gpu_use_mixture,
        run_gpu,
    )


@app.cell
def _(gpu_model_name, mo, new_run_dir, run_gpu, torch):
    mo.stop(not run_gpu.value, mo.md("**GPU 실습 실행** 버튼을 누르면 모델·데이터를 내려받고 학습을 시작합니다."))
    mo.stop(not torch.cuda.is_available(), mo.callout("CUDA GPU가 없어 9부를 실행할 수 없습니다.", kind="warn"))
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

    gpu_output_dir = new_run_dir("decision-internals-gpu")
    # 일반 LM 이름 → decision_from_lm.load_lm_as_decision_model이 새 JointSchemaHead를 붙입니다.
    gpu_model, gpu_tokenizer = FastDecisionModel.from_pretrained(
        model_name=gpu_model_name.value,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    _encoder = sum(_p.numel() for _p in gpu_model.encoder.parameters())
    _head = sum(_p.numel() for _p in gpu_model.head.parameters())
    print(f"결과 저장 위치: {gpu_output_dir}")
    print(type(gpu_model).__name__, "head config:", gpu_model.head.config)
    print("decision_config:", gpu_model.decision_config)
    print(f"백본 매개변수(4비트 저장분 포함) {_encoder / 1e6:.0f}M, 헤드 {_head / 1e6:.1f}M (float32: {next(gpu_model.head.parameters()).dtype})")
    return (
        DecisionTrainer,
        FastDecisionModel,
        gpu_model,
        gpu_output_dir,
        gpu_tokenizer,
        is_bfloat16_supported,
    )


@app.cell
def _(FastDecisionModel, gpu_model):
    gpu_peft_model = FastDecisionModel.get_peft_model(
        gpu_model, r=16, lora_alpha=16, lora_dropout=0, use_gradient_checkpointing="unsloth", random_state=3407
    )
    _trainable = sum(_p.numel() for _p in gpu_peft_model.parameters() if _p.requires_grad)
    print(f"학습 가능 매개변수(LoRA + 헤드): {_trainable / 1e6:.1f}M")
    return (gpu_peft_model,)


@app.cell
def _(FastDecisionModel, gpu_peft_model, gpu_tokenizer, gpu_train_rows, gpu_use_mixture):
    from unsloth.models.decision_datasets import build_decision_mixture, load_source

    _limit = gpu_train_rows.value or None
    if gpu_use_mixture.value:
        _n_rows = _limit or 5000
        gpu_rows = build_decision_mixture({"typed_decisions": 0.8, "boolq": 0.1, "sst5": 0.1}, n_rows=_n_rows, seed=3407)
    else:
        gpu_rows = load_source("typed_decisions", "train", limit=_limit)
    _items, _report = FastDecisionModel.build_dataset(gpu_rows, gpu_tokenizer, gpu_peft_model)
    print(f"{len(gpu_rows)}행 → 레코드 {len(_items)}개, 건너뛴 판단 {_report['skipped']} / {_report['total']}")
    if _report["reason"]:
        print("가장 흔한 건너뜀 이유:", _report["reason"])
    gpu_train_items, gpu_eval_items = FastDecisionModel.split_holdout(_items, seed=3407)
    print(f"학습 {len(gpu_train_items)}행, 홀드아웃 {len(gpu_eval_items)}행")
    gpu_metrics = {"before": FastDecisionModel.evaluate(gpu_peft_model, gpu_tokenizer, gpu_eval_items)}
    print("학습 전 홀드아웃:", gpu_metrics["before"])
    return gpu_eval_items, gpu_metrics, gpu_train_items, load_source


@app.cell
def _(
    DecisionTrainer,
    FastDecisionModel,
    gpu_batch_size,
    gpu_eval_items,
    gpu_metrics,
    gpu_output_dir,
    gpu_peft_model,
    gpu_stage_a_steps,
    gpu_stage_b_steps,
    gpu_tokenizer,
    gpu_train_items,
    is_bfloat16_supported,
):
    from transformers import TrainingArguments


    def _args(name, steps):
        return TrainingArguments(
            per_device_train_batch_size=gpu_batch_size.value,
            gradient_accumulation_steps=4,
            warmup_steps=min(10, max(1, steps // 6)),
            max_steps=steps,
            learning_rate=2e-4,  # LoRA 학습률 (헤드는 head_learning_rate)
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=5,
            output_dir=str(gpu_output_dir / name),
            save_strategy="no",
            report_to="none",
            seed=3407,
        )


    gpu_logs = {}
    if gpu_stage_a_steps.value > 0:
        # 1단계: LoRA 가중치를 고정하고 새 헤드만 학습
        FastDecisionModel.freeze_backbone(gpu_peft_model)
        _trainer = DecisionTrainer(
            model=gpu_peft_model,
            processing_class=gpu_tokenizer,
            train_dataset=gpu_train_items,
            head_learning_rate=1e-3,
            args=_args("stage_a", gpu_stage_a_steps.value),
        )
        gpu_logs["stage_a"] = _trainer.train().metrics
        FastDecisionModel.unfreeze_backbone(gpu_peft_model)
        gpu_metrics["after_stage_a"] = FastDecisionModel.evaluate(gpu_peft_model, gpu_tokenizer, gpu_eval_items)
        print("1단계 후 홀드아웃:", gpu_metrics["after_stage_a"])

    # 2단계: LoRA + 헤드. 6부의 손실 항은 여기서 인자로 켭니다(0이면 교차 엔트로피만).
    _trainer = DecisionTrainer(
        model=gpu_peft_model,
        processing_class=gpu_tokenizer,
        train_dataset=gpu_train_items,
        label_smoothing=0.0,
        brier_weight=0.0,
        ordinal_weight=0.0,
        args=_args("stage_b", gpu_stage_b_steps.value),
    )
    gpu_logs["stage_b"] = _trainer.train().metrics
    gpu_metrics["after_stage_b"] = FastDecisionModel.evaluate(gpu_peft_model, gpu_tokenizer, gpu_eval_items)
    print("2단계 후 홀드아웃:", gpu_metrics["after_stage_b"])
    return (gpu_logs,)


@app.cell
def _(
    FastDecisionModel,
    gpu_eval_items,
    gpu_logs,
    gpu_metrics,
    gpu_model_name,
    gpu_output_dir,
    gpu_peft_model,
    gpu_tokenizer,
    json,
    load_source,
):
    # 8부의 _calibrate_clef: 홀드아웃으로 온도를 맞추고 decision_config에 기록합니다.
    gpu_metrics["calibration"] = FastDecisionModel.calibrate(gpu_peft_model, gpu_tokenizer, gpu_eval_items)
    print("보정(교차 평가):", gpu_metrics["calibration"])
    print("온도:", gpu_peft_model.decision_config.get("head_temperature"), gpu_peft_model.decision_config.get("temperature"))

    _test_items, _ = FastDecisionModel.build_dataset(load_source("typed_decisions", "test"), gpu_tokenizer, gpu_peft_model)
    gpu_metrics["test"] = FastDecisionModel.evaluate(gpu_peft_model, gpu_tokenizer, _test_items)
    print("테스트 분할:", gpu_metrics["test"])

    FastDecisionModel.for_inference(gpu_peft_model)
    gpu_answers = FastDecisionModel.predict(
        gpu_peft_model,
        gpu_tokenizer,
        "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
        {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors", "sales": "pricing, new plans"},
            },
            "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "today"]},
        },
    )
    for _name, _result in gpu_answers.items():
        print(_name, _result["answer"], {_k: round(_v, 3) for _k, _v in _result["probabilities"].items()})

    # LoRA 어댑터, 판단 헤드, 보정 온도가 함께 저장됩니다.
    gpu_peft_model.save_pretrained(str(gpu_output_dir / "decision_lora"))
    (gpu_output_dir / "metrics.json").write_text(
        json.dumps(
            {"model": gpu_model_name.value, "metrics": gpu_metrics, "train": gpu_logs, "answers": gpu_answers},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    print("저장 완료:", gpu_output_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 정리와 연습 문제

    - 판단 모델 = **LM 백본 + 판단 헤드**. 모든 질문을 한 프롬프트로 읽고, 질문마다 선택지별 logit을 냅니다.
    - 프롬프트는 질문·선택지의 **토큰 구간**을 기록하며, 헤드는 그 구간의 은닉 상태(문맥)와 선택지 단어 임베딩(어휘)을 함께 씁니다.
    - 질문 벡터들이 서로 attention하므로 여러 질문을 **공동으로** 판단합니다.
    - 학습은 soft target 교차 엔트로피가 기본이고, 라벨 스무딩·Brier·순서 거리 항을 더할 수 있습니다.
    - 새 헤드는 **헤드만 먼저** 학습한 뒤 백본(LoRA)까지 학습하는 2단계 레시피를 쓸 수 있습니다.
    - 확률은 홀드아웃에서 **온도 스케일링**으로 보정하고 ECE로 확인합니다.

    연습 문제

    1. 4부에서 `team` 선택지 이름을 `A`, `B`, `C`로 바꾸고 설명만 남기면 선택지 순서와 span이 어떻게 바뀌나요?
    2. 5부 `trace_head`에서 decoder 층을 건너뛰면(질문끼리 attention하지 않으면) 어떤 정보를 잃게 되나요?
    3. 6부에서 `label_smoothing=0.2`일 때 "정답 확신" 분포의 손실이 커지는 이유를 설명하세요.
    4. 7부에서 1단계 0스텝 / 2단계 120스텝과 기본 설정의 결과를 비교하세요. 2단계 후 정확도가 비슷하다면, 실제 LLM에서는 왜 1단계가 도움이 될까요?
    5. 7부에서 `refund` 정답이 billing 문의에서만 자주 참입니다. 이 상관관계가 8부 추론에서 어떻게 드러나는지 문장을 바꿔 확인하세요.

    ## 출처와 라이선스

    - Unsloth `decision.py`, `decision_from_lm.py`, `decision_datasets.py`, `_decision_common.py`, `clef.py`:
      [unslothai/unsloth](https://github.com/unslothai/unsloth/tree/main/unsloth/models) (AGPL-3.0).
      이 노트북은 설치된 파일을 그대로 불러오며, `_decision_loss`와 `_clef_decide`의 일부만 옮겨 적었습니다.
    - `joint_schema_model.py`: [Cloudflare Clef](https://huggingface.co/Cloudflare/clef) (Apache-2.0), Unsloth에 수정 없이 포함.
    - 데이터: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0).
      혼합 옵션의 BoolQ(CC-BY-SA-3.0), SST-5는 각 데이터셋의 라이선스를 확인하세요.
    """)
    return


if __name__ == "__main__":
    app.run()
