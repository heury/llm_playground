# RTX 5090 자동 감시·학습 시도 결과

## 결과

학습은 시작되지 않았다. 10분 간격으로 11회 재고를 확인한 뒤 2026-11-10 01:23:54 KST에 EUR-NO-1의 RTX 5090을 확보했으나, SSH 준비 상태 확인이 20분 안에 완료되지 않아 01:44:03에 실패했다. 소스 배포 전 실패이므로 데이터 다운로드, 가중치 결합 검증, 학습, 샘플 생성 및 Hub 업로드는 미실행이다. SSH 확인 실패의 근본 원인은 이번 결과 조회에서 진단하지 않았다.

감시 프로세스는 오류로 종료됐고 자동 재시도는 실행 중이 아니다. 과거 모델 결과를 이번 학습 결과로 사용하지 않았다.

## 리소스 정리

실패 당시 감시 프로세스는 Pod를 정지하고 볼륨을 보존했다. 2026-11-10 결과 확인 때 일반 Pod 목록은 빈 배열이었지만 개별 조회에서 정지 Pod가 확인됐다. 목록만으로 삭제를 판단하면 안 된다.

이번 작업의 Pod `50n8d7t0q3ji9x`를 삭제한 다음 연결된 볼륨 `5mc5vxvpu9`를 삭제했다. 각각 개별 get에서 HTTP404 not_found 응답을 확인했다. 현재 이번 작업의 Pod와 볼륨은 남아 있지 않다. GPU 가격은 생성 시 $1.19/hour였다. 실제 과거 청구액은 조회하지 않았다.

## 환경과 검증 범위

작업 디렉토리: `/home/mind4u/dev/llm_playground`. CLI: runpodctl 3.1.0-5ab53b9. 이미지: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. CUDA 하한12.8, GPU1개, 컨테이너30GB, 네트워크 볼륨50GB, SSH22/tcp. 원본 모델·데이터 리비전은 `projects/minimind-o/scripts/runpod_vlm_audio/prepare.py`에 고정했으나 원격 학습이 시작되지 않아 실제 모델/데이터 로드 검증은 미실행이다.

로컬에서 Python 구문 검사·bash 구문 검사, 재고 부족/할당 성공/인증 실패에 대한 모의 검사 3개만 수행했다. 이 검사는 실제 SSH 배치 성공을 검증하지 않는다.

확인 명령:

```bash
python3 outputs/minimind-o/20261109-vlm-audio/rp.py pod get 50n8d7t0q3ji9x
python3 outputs/minimind-o/20261109-vlm-audio/rp.py network-volume get 5mc5vxvpu9
```

두 명령 모두 정리 후 not_found를 반환했다. 인증 값은 기록하지 않았다.

## 기록

산출물: `outputs/minimind-o/20261109-vlm-audio-auto/`.
- monitor.log: 11회 재고 확인, 할당, SSH 대기, 실패 기록.
- failure-status.json, error.txt: 정리 전 실패와 traceback.
- status.json, resources.json: 실패 및 삭제 확인 후 현재 상태.
- monitor-checks.json: 모의 검사 3개 결과.
- source.tar.gz, source.json: 준비했던 소스와 번들 해시.

실행 명령은 `/tmp/minimind-v-exercise-venv/bin/python -u projects/minimind-o/scripts/runpod_vlm_audio/monitor.py --interval 600 --wait-hours 24`였다. 현재는 종료된 상태다.
