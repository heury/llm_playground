# 영어 강아지 소리 질문

대상: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`. 실행 때 실제 리비전을 기록한다.

각 질문을 대화 이력과 시스템 프롬프트 없이 독립 입력한다.

1. What sound does a dog make?
2. What sound does a puppy make?
3. How do you write the sound of a dog barking in English?

모델 카드에 따라 chat template의 빈 thinking block을 제거한다. CPU float32,
greedy (`do_sample=False`, temperature=1, top_p=1, top_k=0,
repetition_penalty=1), max_new_tokens=128로 실행한다. 총 길이는 340 이하로 유지한다.

판정: 1·2번은 bark/woof/ruff 또는 적절한 whine/yelp 등 실제 발성을
질문에 직접 연결해 답하는지 검토한다. 3번은 woof/ruff/arf 등 적절한 영어
의성어를 제시하는지 확인한다. 질문에 이미 포함된 barking을 반복하는 것만으로
통과시키지 않는다. 무관한 설명, 반복, 잘못된 의성어와 토큰 상한 도달을 기록한다.
토큰 상한에 도달하면 생성된 범위만 평가하고 완성된 답변으로 취급하지 않는다.

실제 응답·토큰·설정은 `outputs/minimind/<실행 ID>/`, 보고서는
`reports/minimind/<실행 ID>.md`에 보관한다.
