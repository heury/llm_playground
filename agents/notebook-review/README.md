# 실습 노트북 평가·업데이트

전체 절차는 **[notebook-review Skill](../../skills/notebook-review/SKILL.md)**에서 관리합니다.
공통 계약·판정·수정·검증·재평가는 Skill이, 독립적인 관점별 판단은 평가 Agent가 담당합니다.

| 구성 | 위치 | 역할 |
| --- | --- | --- |
| Skill | [SKILL.md](../../skills/notebook-review/SKILL.md) | 평가·업데이트 진입점 |
| 공통 기준 | [rubric.md](../../skills/notebook-review/references/rubric.md) | 항목 집계·점수·사용 판정 |
| 운영 절차 | [workflow.md](../../skills/notebook-review/references/workflow.md) | 조건 고정 → 평가 → 수정 → 검증 → 재평가 |
| 입력 형식 | [result-format.md](../../skills/notebook-review/references/result-format.md) | 계약·평가 JSON 형식 |
| 계산·검사 | [review.py](../../skills/notebook-review/scripts/review.py) | 계약·소스 해시 검사, 역할별 점수 계산 |

기존 TOML은 아래 역할 정의를 읽는 연결점입니다. 지표를 중복 보관하지 않습니다.

| 관점 | 기존 호출 | 실제 지표 정의 |
| --- | --- | --- |
| 학생 | [student.toml](student.toml), `notebook_student` | [student.toml](../../skills/notebook-review/references/roles/student.toml) |
| 선생님 | [teacher.toml](teacher.toml), `notebook_teacher` | [teacher.toml](../../skills/notebook-review/references/roles/teacher.toml) |
| 편집자 | [author.toml](author.toml), `notebook_author` | [author.toml](../../skills/notebook-review/references/roles/author.toml) |
| 연구자 | [researcher.toml](researcher.toml), `notebook_researcher` | [researcher.toml](../../skills/notebook-review/references/roles/researcher.toml) |
| 통합 | [coordinator.toml](coordinator.toml), `notebook_coordinator` | Skill의 절차를 실행 |

각 평가 관점은 5개 지표·20개 항목·100점을 유지합니다(지표 버전 2.0).
편집자의 E5는 현재 설명의 완전성입니다. 편집 후 의미 보존은 diff로 별도 확인합니다.
선생님은 projects/를 포함한 실제 참조 구현을 비교합니다. 연구자는 고정한 필수 이론을 기준으로 평가합니다.

## 사용

설치된 Skill을 사용할 때:

```text
$notebook-review notebooks/minimind/minimind_structure.py를 네 관점으로 평가하고 업데이트해줘.
독자는 Python과 기초 텐서를 아는 LLM 입문자야.
작은 CPU 검증과 marimo 검사는 수행하되 다운로드·설치·전체 학습·클라우드 생성은 하지 마.
수정 후 동일 기준으로 재평가하고 남은 문제와 검증 범위를 보고해줘.
```

평가만 필요하면 “평가만 해줘”, 작은 변경이면 “학생·편집자 관점으로 해당 절만 검토해줘”라고 요청합니다.
부분 검토는 네 역할 전체 채점과 구분합니다.
설치 없이도 `skills/notebook-review/SKILL.md를 읽고 …`라고 경로를 지정할 수 있습니다.

원본은 `skills/notebook-review/`입니다. 사용자 설치 위치는 `~/.codex/skills/notebook-review/`이며
현재 세션에 새 Skill이 표시되지 않으면 새 세션에서 확인합니다. 저장소 원본을 바꾸면 설치 사본도
갱신해야 합니다. 이 Skill은 저장소 내부 AGENTS.md와 프로젝트별 실행 환경을 따릅니다.

## 기존 Agent 등록 유지

기존 TOML을 이미 등록했다면 파일을 비교해 갱신합니다. 신규 등록은 쓰기 가능한 환경에서
저장소 루트 기준으로 아래 명령을 사용합니다. `cp -n`은 기존 파일을 갱신하지 않습니다.

```bash
mkdir -p .codex/agents
cp -n agents/notebook-review/student.toml .codex/agents/notebook_student.toml
cp -n agents/notebook-review/teacher.toml .codex/agents/notebook_teacher.toml
cp -n agents/notebook-review/author.toml .codex/agents/notebook_author.toml
cp -n agents/notebook-review/researcher.toml .codex/agents/notebook_researcher.toml
cp -n agents/notebook-review/coordinator.toml .codex/agents/notebook_coordinator.toml
```

연결점은 이 저장소의 Skill 경로를 사용합니다. 독립적으로 배포할 때는 Skill 묶음을 사용합니다.
현재 환경의 프로젝트 `.codex/`는 읽기 전용이므로 Agent 자동 등록은 하지 않습니다.
Skill 자체가 필요한 역할을 subagent에 전달할 수 있으므로 Agent 사전 등록은 필수가 아닙니다.

## 검증과 산출물

```bash
# 저장소 루트에서 실행. 로그는 실행별 outputs/ 경로에 보관합니다.
python3 -m unittest discover -s skills/notebook-review/tests -v
```

계약 템플릿은 [contract-template.json](contract-template.json), 재사용 시나리오는
[테스트 시나리오](../../test_scenarios/notebook-review/README.md)를 참고합니다.
계약·전후 평가·수정 목록·한계는 `reports/<프로젝트>/<실행 ID>-*`,
기준선·diff·해시·로그는 `outputs/<프로젝트>/<실행 ID>/`에 저장합니다.
기존 MiniMind 실습의 프로젝트명은 `notebooks`입니다.
점수는 설명 충족도이며 실제 학생 이해나 실행 성공을 증명하지 않습니다.
