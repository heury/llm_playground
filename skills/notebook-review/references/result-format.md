# 계약·평가 JSON 입력 형식

모든 명령의 스크립트 경로는 Skill 위치 기준이며 `--root`는 평가 대상 저장소입니다.
스코어러는 데이터를 읽기만 하고 파일을 수정하거나 학습 코드를 실행하지 않습니다.

## 계약

`assets/contract-template.json`을 복사해 workflow.md의 필드를 채웁니다.
`template_only=false`, 지표 버전 `2.0`, 80개 항목과 적용/제외·핵심 여부가 필요합니다.
전체 평가에서는 네 역할 모두의 항목을 채웁니다. 핵심 항목은 하나 이상 지정합니다.
필수 이론은 목표에 필요한 개념 목록이며 입문자의 선수 지식과 구분합니다.
`environment`는 환경 설명 문자열, `class_minutes`는 양의 정수 또는 미지정 null입니다.

참조 구현의 종류:

- `kind=local`(기본): `{path, sha256, symbols: ["Attention"], reason}`.
  대상 파일과 별개로 존재하는 실제 구현 파일의 전체 SHA-256입니다.
- `kind=embedded`: `{kind, path, sha256, symbols: ["Attention", "train"], reason}`.
  노트북 안의 독립 구현입니다. `path`는 대상 경로이며 sha256은 전체 파일이 아닌 지정 심볼의 AST 해시입니다.
  `review.py`의 `embedded_fingerprint(Path(...), symbols)`로 계산합니다. 단독 심볼 이름은
  파일에서 유일해야 하고, 클래스 메서드는 `Attention.forward`처럼 지정할 수 있습니다.
  설명만 바꾸면 참조 해시가 유지되고 모델 구현을 바꾸면 새 계약·기준선이 필요합니다.
  심볼 밖의 상수·도움 함수도 모델에 영향을 주면 심볼/별도 참조 목록과 검토 범위에 포함하세요.
- `kind=external`: `{kind, source, revision, symbols, reason}`.
  공식 소스 URL과 패키지 버전/커밋을 기록합니다. 스크립트는 인터넷에서 내용을 확인하지 않으므로
  평가자가 실제 열람해 검증하고 평가 JSON의 `references_verified`에 반영해야 합니다.

`after_files`에는 수정본의 `{path, sha256}` 목록을 넣습니다. 기준선과 같은 경로여야 합니다.
새 보조 파일 등 평가 대상 범위를 변경하면 계약 버전을 올려 새 기준선으로 평가합니다.
그림 등 별도 수정·검증 파일을 모두 점수 평가 대상으로 지정할 필요는 없습니다.

## 역할별 평가 JSON

다음은 일부 항목만 보여주는 형식 예시입니다. 실제 입력에는 해당 역할의 20개 항목을 모두 넣습니다.

```json
{
  "contract_id": "sampling-v1",
  "contract_version": 1,
  "rubric_version": "2.0",
  "role": "student",
  "phase": "baseline",
  "target_hashes": {"notebooks/example.py": "파일의 SHA-256"},
  "references_verified": true,
  "checkpoints": [
    {
      "id": "S1.1",
      "units": [
        {"id": "intro", "status": "pass", "evidence": "example.py:15 목표와 선수 지식 명시"}
      ]
    }
  ],
  "findings": [
    {
      "id": "S-01",
      "severity": "major",
      "status": "open",
      "checkpoints": ["S2.1"],
      "evidence": "example.py:80 입력 shape 설명 누락"
    }
  ]
}
```

- `role`: `student`, `teacher`, `author`(편집자), `researcher`.
- `phase`: `baseline` 또는 `after`. `target_hashes`는 해당 단계의 파일 해시와 정확히 일치해야 합니다.
- 항목의 `units`에는 계약의 적용 단위를 빠짐없이 넣습니다. 적용 제외 항목은 `units=[]`입니다.
- 단위 상태는 `pass`, `fail`, `unknown`입니다. 모든 상태에 근거 위치 또는 미확인 이유가 필요합니다.
- 심각도는 `critical`, `major`, `minor`, 발견 상태는 `open`, `partial`, `resolved`입니다.
  기준선의 미해결 문제는 open, 수정 후 부분 해결은 partial로 남깁니다. findings는 없으면 []입니다.
- 핵심 항목 fail은 발견 목록 누락과 무관하게 수정 후 사용입니다. 핵심 단위 unknown은
  다른 단위의 fail 때문에 집계 상태가 fail이어도 미확인 정보가 보존됩니다.
- JSON은 채점 근거와 산식 입력입니다. 수정안·완료 조건·장점·발견 처리와 실제 검증 결과는
  함께 작성하는 보고서에 기록합니다. 스크립트가 출력하는 `execution_verified=false`는
  계산기가 실행 검증을 수행하지 않았다는 뜻이며 실제 검사 결과를 대체하지 않습니다.

```bash
python3 <Skill>/scripts/review.py validate-contract <계약.json> --root <저장소>
python3 <Skill>/scripts/review.py score <계약.json> <평가.json> --root <저장소>
python3 <Skill>/scripts/review.py validate-contract <계약.json> --phase after --root <저장소>
```

오류는 종료 코드 2와 stderr로 반환됩니다. 결과 JSON은 stdout으로 출력하므로
`outputs/<프로젝트>/<실행 ID>/`의 파일로 리다이렉트해 보관합니다.
확인률은 0~1, 점수는 반올림 전 숫자입니다. 보고서에서는 확인률을 %로, 점수를 소수 첫째 자리로 표시합니다.
