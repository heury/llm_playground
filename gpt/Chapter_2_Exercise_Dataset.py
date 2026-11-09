import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter_2_Exercise_Dataset

    원본 `Chapter_2_Exercise_Dataset.ipynb`의 설명·모델·학습 설정을 유지한 marimo 변환본입니다.
    `gpt/`에서 `uv sync --locked` 후 `uv run marimo edit "Chapter_2_Exercise_Dataset.py"`로 실행하세요.
    데이터·토크나이저·사전 학습 가중치는 첫 실행에 다운로드될 수 있습니다.
    5~7장의 전체 학습에는 충분한 메모리와 GPU 실행 환경을 권장합니다.
    실행 산출물과 데이터는 `outputs/gpt/<실행 ID>/`, 재사용 가중치는 `gpt/models/gpt2/`에 저장합니다.
    이번 변환에서는 전체 학습을 실행하지 않았습니다.
    셀을 수정하면 의존 셀이 다시 실행되므로 학습 셀은 marimo의 지연 실행 모드에서 실행을 관리하세요.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from marimo_runtime import prepare_run
    output_dir = prepare_run('chapter_2_exercise_dataset')
    mo.md(f"실행 산출물: `{output_dir}`")
    return mo, output_dir


@app.cell
def _():
    import importlib
    import tiktoken
    print('tiktoken version:', importlib.metadata.version('tiktoken'))
    tokenizer = tiktoken.get_encoding('gpt2')
    _text = 'Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.'
    integers = tokenizer.encode(_text, allowed_special={'<|endoftext|>'})
    print(integers)
    strings = tokenizer.decode(integers)
    # region  [token encoding]
    # endregion
    # region  [token decoding]
    print(strings)
    return tiktoken, tokenizer


@app.cell
def _(output_dir, tokenizer):
    import os
    import requests

    os.makedirs(f"{output_dir}/datas", exist_ok=True)
    if not os.path.exists(f"{output_dir}/datas/the-verdict.txt"):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        file_path = f"{output_dir}/datas/the-verdict.txt"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
    with open(f"{output_dir}/datas/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    enc_sample = enc_text[50:]

    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"x: {x}")
    print(f"y:      {y}")
    return (raw_text,)


@app.cell
def _():
    ### 중요: 이 코드는 GPT 모델 학습을 위한 데이터셋 클래스를 정의합니다

    import torch
    from torch.utils.data import Dataset, DataLoader

    class GPTDatasetV1(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
            """
            Args:
                txt (str): 학습할 전체 텍스트 데이터
                tokenizer: 텍스트를 토큰 ID로 변환해주는 토크나이저 (예: tiktoken)
                max_length (int): 모델이 한 번에 볼 수 있는 윈도우 크기 (입력 시퀀스 길이)
                stride (int): 윈도우를 이동시킬 간격 (데이터 중복 정도를 결정)
            """
            self.input_ids = []
            self.target_ids = []

            # 1. 전체 텍스트 토큰화
            # 텍스트를 정수 리스트(token_ids)로 변환합니다.
            # <|endoftext|> 같은 특수 토큰도 허용하여 인코딩합니다.
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

            # 데이터가 너무 짧으면 학습할 수 없으므로 최소 길이를 확인합니다.
            assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."

            # 2. 슬라이딩 윈도우(Sliding Window)로 데이터 생성
            # 전체 토큰 리스트를 훑으며 max_length 길이만큼 잘라냅니다.
            # stride만큼 건너뛰며 반복합니다.
            for i in range(0, len(token_ids) - max_length, stride):
                # 입력 청크: 현재 위치(i)부터 max_length만큼 가져옵니다.
                # region [input 청크 생성]
                input_chunk = token_ids[i : i + max_length]
                # endregion
              
                # 타겟 청크: 입력보다 1칸 뒤의 위치(i+1)부터 가져옵니다.
                # GPT는 '다음 단어'를 맞추는 모델이므로, 정답은 입력보다 한 칸씩 뒤로 밀려있어야 합니다.
                # region [target 청크 생성]
                target_chunk = token_ids[i + 1 : i + max_length + 1]
                # endregion

                # 추출한 데이터를 텐서(Tensor)로 변환하여 리스트에 저장합니다.
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            # 데이터셋의 총 샘플(청크) 개수를 반환합니다.
            return len(self.input_ids)

        def __getitem__(self, idx):
            # DataLoader가 데이터를 요청할 때 호출됩니다.
            # 해당 인덱스(idx)의 입력과 정답 쌍을 반환합니다.
            return self.input_ids[idx], self.target_ids[idx]

    return DataLoader, GPTDatasetV1, torch


@app.cell
def _(DataLoader, GPTDatasetV1, tiktoken):
    ##중요 : 아래 함수는 DataLoader를 생성하는 헬퍼 함수입니다.
    def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        tokenizer = tiktoken.get_encoding('gpt2')
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader  # 토크나이저를 초기화합니다.  # 데이터셋을 만듭니다.  # 데이터 로더를 만듭니다.  # region [DataLoader 생성]  # endregion

    return (create_dataloader_v1,)


@app.cell
def _(create_dataloader_v1, raw_text):
    _dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    _data_iter = iter(_dataloader)
    first_batch = next(_data_iter)
    print(first_batch)
    second_batch = next(_data_iter)
    print(second_batch)
    return


@app.cell
def _(torch):
    input_ids = torch.tensor([2, 3, 5, 1])
    _vocab_size = 6
    _output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    print(embedding_layer.weight)
    # region [임베딩 레이어 생성]
    _token_embeddings = embedding_layer(input_ids)
    # endregion
    # region [임베딩 레이어 적용]
    print(_token_embeddings)
    return


@app.cell
def _(create_dataloader_v1, raw_text, torch):
    _vocab_size = 50257
    _output_dim = 256
    token_embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    max_length = 4
    _dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    _data_iter = iter(_dataloader)
    inputs, targets = next(_data_iter)
    _token_embeddings = token_embedding_layer(inputs)
    print(_token_embeddings.shape)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, _output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)
    input_embeddings = _token_embeddings + pos_embeddings
    # region [포지셔널 임베딩 레이어 생성]
    # endregion
    # region [포지셔널 임베딩 적용]
    # region [토큰 임베딩과 포지셔널 임베딩 합산]
    print(input_embeddings.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokenizer 비교
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 아래 결과를 보면 한국어는 Exaone, 중국어는 Qwen의 Tokenizer 성능이 좋습니다
    """)
    return


@app.cell
def _():
    from transformers import AutoTokenizer
    # Tokenizer 로드 (로컬 또는 온라인에서)
    print('Tokenizer 로드 중...')
    qwen = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    exaone = AutoTokenizer.from_pretrained('LGAI-EXAONE/EXAONE-4.0-1.2B')
    texts = ['안녕하세요, 반갑습니다!', 'Hello, nice to meet you!', '인공지능 기술이 발전하고 있습니다.', 'Hello! This is an English tokenizer test.', 'こんにちは！日本語のテストです。', '人工智能技术正在快速发展。', 'The quick brown fox jumps over the lazy dog. 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.', '파이썬 프로그래밍은 매우 유용합니다. Python programming is very useful.', '123456789 !@#$%^&*() 특수문자']
    # 테스트할 텍스트
    print('\n' + '=' * 60)
    for _text in texts:
        print(f'\n텍스트: {_text}')
        print('-' * 60)
        qwen_tokens = qwen.encode(_text, add_special_tokens=True)
        qwen_token_strs = [qwen.decode([tok]) for tok in qwen_tokens[:10]]
        print(f'Qwen3:  {len(qwen_tokens)}개 토큰')
        print(f'        {qwen_token_strs}')
        exaone_tokens = exaone.encode(_text, add_special_tokens=True)
        exaone_token_strs = [exaone.decode([tok]) for tok in exaone_tokens[:10]]
        print(f'EXAONE: {len(exaone_tokens)}개 토큰')
        print(f'        {exaone_token_strs}')
        diff = len(qwen_tokens) - len(exaone_tokens)
        if diff > 0:
            print(f'→ EXAONE이 {diff}개 더 효율적')
        elif diff < 0:
            print(f'→ Qwen3가 {abs(diff)}개 더 효율적')
        else:  # Qwen3
            print(f'→ 동일')
    print('\n' + '=' * 80)
    print('4. Special Tokens 비교')
    print('=' * 80)
    print(f'\n[Qwen3 Special Tokens]')
    for token_name in ['bos_token', 'eos_token', 'unk_token', 'pad_token']:  # EXAONE
        token = getattr(qwen, token_name, None)
        if token:
            token_id = getattr(qwen, f'{token_name}_id', None)
            print(f"  {token_name}: '{token}' (ID: {token_id})")
    print(f'\n[EXAONE Special Tokens]')
    for token_name in ['bos_token', 'eos_token', 'unk_token', 'pad_token']:  # 비교
        token = getattr(exaone, token_name, None)
        if token:
            token_id = getattr(exaone, f'{token_name}_id', None)
            print(f"  {token_name}: '{token}' (ID: {token_id})")  # Special tokens 비교
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hugging Face 모델 파일 구조 설명

    Hugging Face Hub에 업로드된 파일들은 **Transformer 기반 언어 모델(LLM)**을 구성하는 핵심 요소들입니다. 크게 **설계도(Config)**, **가중치(Weights)**, **토크나이저(Tokenizer)** 세 그룹으로 나뉩니다.

    ---

    ### 1. 모델 아키텍처 및 설정 (설계도)

    * **`config.json`**
        * **의미:** 모델의 **구조와 설계도**가 담긴 가장 중요한 설정 파일입니다.
        * **내용:** 모델의 종류(Llama, BERT 등), 레이어(Layer) 수, 히든 사이즈, 어텐션 헤드 수 등을 정의하여 빈 모델(Skeleton)을 만듭니다.

    * **`generation_config.json`**
        * **의미:** 텍스트 **생성(Generation) 시 사용할 기본 옵션** 파일입니다.
        * **내용:** `max_length`(최대 길이), `temperature`(창의성), `top_p`, `eos_token_id`(종료 토큰) 등의 기본 설정값이 들어있습니다.

    ### 2. 모델 가중치 (학습된 뇌)

    * **`model.safetensors`**
        * **의미:** 실제 **학습된 파라미터(가중치)**가 저장된 파일입니다. (가장 용량이 큼)
        * **특징:**
            * 기존 PyTorch의 `.bin` 파일보다 로딩 속도가 빠릅니다.
            * 보안 취약점(악성 코드 실행 등)을 해결한 안전한 형식입니다.

    ### 3. 토크나이저 (언어 처리기)
    사람의 언어(Text)를 모델이 이해하는 숫자(Token ID)로 변환하는 규칙들입니다.

    * **`vocab.json`**
        * **의미:** **단어장(Vocabulary)** 파일입니다. 단어와 그에 대응하는 숫자 ID가 매핑되어 있습니다.
    * **`merges.txt`**
        * **의미:** **BPE 병합 규칙** 파일입니다. 알파벳 조각들이 어떻게 합쳐져서 단어가 되는지 정의합니다.
    * **`tokenizer.json`**
        * **의미:** `vocab.json`과 `merges.txt` 등을 하나로 통합하여 최적화한 파일로, 로딩 속도가 매우 빠릅니다.
    * **`tokenizer_config.json`**
        * **의미:** 토크나이저 설정 파일입니다. 어떤 클래스를 쓸지, 특수 토큰(`<s>`, `<pad>` 등)은 무엇인지 정의합니다.

    ---

    ### 요약

    | 파일명 | 비유 | 역할 |
    | :--- | :--- | :--- |
    | `config.json` | **설계도** | 모델 구조 정의 |
    | `model.safetensors` | **뇌** | 학습된 지식(가중치) 저장 |
    | `tokenizer.json` | **번역기** | 텍스트 ↔ 숫자 변환 |
    | `generation_config.json` | **말하기 습관** | 생성 옵션 설정 |
    """)
    return


if __name__ == "__main__":
    app.run()
