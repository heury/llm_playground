import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter_1_Exercise_Vector Space

    원본 `Chapter_1_Exercise_Vector Space.ipynb`의 설명·모델·학습 설정을 유지한 marimo 변환본입니다.
    `gpt/`에서 `uv sync --locked` 후 `uv run marimo edit "Chapter_1_Exercise_Vector Space.py"`로 실행하세요.
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
    output_dir = prepare_run('chapter_1_exercise_vector-space')
    mo.md(f"실행 산출물: `{output_dir}`")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gensim 실습 예제
    아래 코드는 `gensim`으로 단어 벡터를 만들고, 의미 계산/유사 단어 탐색/단어 그룹 분류를 수행합니다.
    """)
    return


@app.cell
def _():
    import gensim.downloader as api

    def run_gensim_tutorial():
        # 1. 사전 학습된 가벼운 모델 다운로드 및 로드 (약 65MB)
        # 강의 중 첫 실행 시 다운로드 시간이 약간 소요될 수 있습니다.
        print("모델을 로딩 중입니다... (glove-wiki-gigaword-50)\n")
        model = api.load("glove-wiki-gigaword-50")
    
        # ----------------------------------------------------
        # [기능 1] 단어의 의미(유사도) 계산하기
        # 두 단어 벡터 간의 코사인 유사도(Cosine Similarity)를 계산합니다.
        # ----------------------------------------------------
        print("1. 단어 간 유사도 계산하기")
        word1, word2, word3 = 'cat', 'dog', 'car'
    
        sim_cat_dog = model.similarity(word1, word2)
        sim_cat_car = model.similarity(word1, word3)
    
        print(f" - '{word1}'와 '{word2}'의 유사도: {sim_cat_dog:.4f}")
        print(f" - '{word1}'와 '{word3}'의 유사도: {sim_cat_car:.4f}\n")
    
        # ----------------------------------------------------
        # [기능 2] 유사한 단어 가져오기
        # 특정 단어와 벡터 공간상에서 가장 가까운 단어들을 추출합니다.
        # ----------------------------------------------------
        print("2. 'computer'와 가장 유사한 단어 5개 가져오기")
        target_word = 'computer'
        similar_words = model.most_similar(target_word, topn=5)
    
        for word, score in similar_words:
            print(f" - {word} (유사도 점수: {score:.4f})")
        print()
    
        # ----------------------------------------------------
        # [기능 3] 주어진 단어들을 그룹으로 분류 (이질적인 단어 찾기)
        # 단어들의 의미적 군집을 파악하여, 그룹에 어울리지 않는 단어를 찾아냅니다.
        # ----------------------------------------------------
        print("3. 단어 그룹 중 성격이 다른 단어(Outlier) 분류하기")
        word_group = ['apple', 'banana', 'orange', 'car']
    
        outlier = model.doesnt_match(word_group)
        print(f" - 단어 그룹: {word_group}")
        print(f" - 과일 그룹에 어울리지 않는 단어: '{outlier}'\n")

    # 실행
    run_gensim_tutorial()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Online 실습 예제
    백터공간에서 단어 의미를 시뮬레이션합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/
    - Word Embedding Demo
    - Soldier는 어느 위치일까?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - https://vectors.nlpl.eu/explore/embeddings/en/calculator/
    - Korea_PROPN, Busan_PROPN 을 입력한 후 Japan_PROPN을 입력하면 뭐가 나올까요?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - https://vectors.nlpl.eu/explore/embeddings/en/associates/#
    - mother을 입력하면 연관된 단어는 어떤 것이 나올까요?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - https://vectors.nlpl.eu/explore/embeddings/en/visual/
    - car,tank,transport,computer,mouse,Moscow,Paris,London,Russia,France,Britain,clean,dirty,new 를 입력하면 어떻게 분류가 될까요?
    """)
    return


if __name__ == "__main__":
    app.run()
