# 배터리 이미지 데이터 전처리 파이프라인 v3.9

v3.9은 Battery ID 단위 누수 방지 분할, 결정론적 실행, YOLO detection/segmentation 변환과 승인 후 staging 승격 구조를 유지하면서 CT 대형 bbox 제외 정책을 추가합니다.

## 핵심 정책과 처리 순서

```text
raw scan 및 구조 검증
→ polygon 복구·ROI clipping·multipart 분리
→ 8자리 YOLO 정규화 좌표로 개별 porosity bbox 면적비 계산
→ 이미지별 porosity_bbox_max_ratio 계산
→ porosity_bbox_max_ratio >= 0.25 CT 이미지 선제 제외
→ 제외 후 Battery ID 통계 재계산
→ CT Test 7 ID / development 40 ID 선정
→ development 5-fold 층화
→ 품질 보고서·검토·승인
→ execute 재계산 및 최종 출력
```

제외 조건은 경계를 포함합니다.

```python
sample.modality == "CT" and sample.porosity_bbox_max_ratio >= 0.25
```

- 기준은 polygon 면적합이 아니라 개별 유효 porosity 조각 bbox 면적비의 최댓값입니다.
- bbox 폭과 높이는 실제 YOLO 출력과 동일한 소수점 8자리 정규화 값으로 계산합니다.
- `porosity_area_sum_ratio`와 면적 구간은 EDA와 fold 층화에만 사용합니다.
- RGB/EXT에는 이 정책을 적용하지 않습니다.
- 원본 파일은 수정하거나 삭제하지 않습니다.
- 한 CT ID의 모든 이미지가 제외되어 잔여 CT ID가 47개 미만이면 구조 게이트에서 중단합니다.

## 주요 보고서

| 파일 | 내용 |
|---|---|
| `ct_bbox_exclusions.csv` | bbox 최대 면적비 25% 이상으로 제외된 CT 이미지 전수 |
| `manifest.csv` | 전체 이미지 lineage, bbox 최대 비율, 선택·제외 상태와 사유 |
| `id_scan_report.csv` | ID별 정책 적용 전후 이미지 장수와 통계 |
| `selected_battery_ids_candidate.csv` | 제외 후 통계로 계산한 승인 전 ID 배정 |
| `ct_split_area_distribution.csv` | 선택 후 fold/Test의 축×면적합 구간 분포 |
| `dryrun_warnings.csv` | §17.2 승인 필수 품질 경고 |
| `quality_exceptions.csv` | 품질 경고 예외 승인과 감사 필드 |
| `review_warnings.csv` | §17.3 검토 경고; 자동 실행 차단 대상은 아님 |
| `split_id_leakage_audit.csv` | split/fold 간 ID 및 pixel hash 누수 감사 |

고정 제외 사유는 다음과 같습니다.

```text
ct_porosity_bbox_max_ratio_ge_0.25
```

## 품질 예외 승인

`dryrun_warnings.csv`가 비어 있지 않으면 대응하는 `quality_exceptions.csv` 행에 다음 값을 모두 기록해야 승인할 수 있습니다.

```text
status=approved_exception
reviewer=<검토자>
reviewed_at=<검토 시각>
reason=<승인 사유>
```

`warning_id`는 경고 코드·관측값·임계값에서 생성한 안정적인 SHA-256 키입니다. execute는 경고 ID와 관측값이 승인 시점과 같은지 다시 확인합니다. 구조·누수 게이트에는 예외 승인을 적용하지 않습니다.

## 설치

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
```

## 실행

`run_pipeline.ps1`이 네 단계의 단일 진입점입니다. 모듈 이름을 폴더에서 찾으므로 버전이 올라가도 스크립트를 고칠 필요가 없습니다. **pwsh(PowerShell 7) 전용**입니다. Windows PowerShell 5.1은 BOM 없는 UTF-8을 ANSI로 읽어 한글 경로에서 깨집니다.

```powershell
pwsh -File .\run_pipeline.ps1 -Stage dry-run `
  -RawRoot "<raw 데이터 루트>" -WorkDir "<v3.9 작업 폴더>" -Jobs 8 -Detached

pwsh -File .\run_pipeline.ps1 -Stage approve `
  -WorkDir "<v3.9 작업 폴더>" -ApprovedBy "<검토자>"

pwsh -File .\run_pipeline.ps1 -Stage execute `
  -RawRoot "<raw 데이터 루트>" -WorkDir "<v3.9 작업 폴더>" `
  -Output "<최종 출력 폴더>" -Jobs 8 -Detached

pwsh -File .\run_pipeline.ps1 -Stage upload `
  -WorkDir "<v3.9 작업 폴더>" -Output "<최종 출력 폴더>" `
  -Remote "gdrive:<업로드 경로>" -Detached
```

`-Detached`는 스크립트를 독립 프로세스로 다시 띄웁니다. dry-run과 execute는 각각 약 37분과 100분이 걸리는데, 에이전트 세션의 자식 프로세스는 툴 호출이 끝나면 정리되므로 이 스위치 없이 돌리면 로그도 남기지 못한 채 사라집니다.

진행 상황은 `<작업 폴더>\pipeline.status`에서 확인합니다. 마지막 줄이 `finished ... exit=0`이면 정상 완료입니다.

execute는 raw fingerprint, seed, 선제 제외 정책, 제외 후 ID 통계, 승인된 ID 배정, 품질 예외와 검토 경고 집합을 다시 검증합니다. upload는 각 zip의 크기가 안정되고 중앙 디렉터리가 읽힐 때만 전송하므로 미완성 아카이브가 올라가지 않습니다.

## 선정 로직 검증

선정 로직을 고칠 때 37분짜리 dry-run을 반복하지 않아도 됩니다.

```powershell
.\.venv\Scripts\python.exe tools\verify_ct_selection.py "<작업 폴더>\reports" --check
```

완료된 dry-run의 `manifest.csv`에서 CT 샘플을 복원해 파이프라인이 쓰는 선정 함수를 그대로 호출하고, 약 90초 만에 fold 편차와 Test/development 균형을 보여줍니다. `--check`는 산출 결과가 기록된 배정과 일치하는지 확인합니다. **코드를 고치기 전에 이 검사를 먼저 통과시키십시오.** 통과해야만 이 도구의 측정값을 근거로 쓸 수 있습니다.

## 테스트

```powershell
python -m compileall -q battery_v3_9 tests
python -m unittest discover -s tests -v
```

v3.6의 work-dir, 승인 파일 또는 ID CSV를 재사용하지 마십시오. v3.9 전용 작업 폴더에서 dry-run부터 다시 실행해야 합니다.
