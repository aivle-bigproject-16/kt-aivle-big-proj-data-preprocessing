# 배터리 데이터 전체 전처리 v3.6

이 폴더가 전체 전처리 실행 코드다. v3.5 전체 파이프라인에 CT porosity 면적합 층화를 추가했으며, v3.4와 충돌하는 항목은 최신 결정인 v3.5를 적용한다.

## v3.6 추가 사항

- CT 이미지의 유효 porosity 폴리곤 면적합을 ROI 면적으로 나눈다.
- porosity가 없는 정상 CT도 `zero` 면적 구간에 포함한다.
- 기존 `defect_image_ratio` 1차 균형과 `annotations_per_image` ±20% 게이트를 유지한다.
- development 40 ID를 5개 fold에 정확히 8 ID씩 배정하면서 이미지 장수, x/y/z, 면적 구간, 축×면적 구간을 함께 맞춘다.
- 기존 Test 7 ID 승인본이 있으면 그대로 잠근다. 신규 Test 선정 때는 v3.5의 축·결함비 우선순위를 유지하고 면적 구간 차이를 마지막 보조 지표로 사용한다.
- `manifest.csv`에 `porosity_polygon_count`, `porosity_area_sum_ratio`, `porosity_area_bin`, `ct_stratum`을 추가한다.
- `ct_split_area_distribution.csv`, `ct_split_balance_summary.csv`, `ct_large_area_review.csv`를 추가한다.

## 전체 실행

`--raw-root`는 `Training`과 `Validation`의 원천·라벨 폴더를 포함하는 `3.개방데이터\1.데이터` 위치로 지정한다. 코드는 그 아래를 재귀 탐색한다.

```powershell
Set-Location "C:\Users\User\Downloads\최종 v3.6 구현\battery_v3_6_preprocessor"

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock

.\.venv\Scripts\python.exe -m battery_v3_6.cli dry-run `
  --raw-root "E:\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터" `
  --work-dir "E:\battery_v3_6_work" `
  --jobs 8

.\.venv\Scripts\python.exe -m battery_v3_6.cli approve-selection `
  --work-dir "E:\battery_v3_6_work" `
  --approved-by "검토자"

.\.venv\Scripts\python.exe -m battery_v3_6.cli execute `
  --raw-root "E:\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터" `
  --work-dir "E:\battery_v3_6_work" `
  --output "E:\battery_v3_6_output" `
  --jobs 8
```

`dry-run → approve-selection → execute` 순서는 계획서의 승인 게이트이므로 생략하지 않는다. 최종 출력은 CT/RGB 이미지, 원본 JSON 보존본, YOLO detection/segmentation 라벨, fold/split 목록, manifest, 보고서와 ZIP을 포함한다.

## 전체 데이터의 범위

전체 raw를 검사하지만 원본을 그대로 전부 복제하지는 않는다. 계획서가 정의한 최종 학습 대상 전체를 출력한다.

- CT: Test 7 ID와 development 40 ID의 모든 유효 이미지
- RGB: 대표 160 ID. Train/Validation은 ID별 최대 300장, Test는 전량
- 제외 데이터: 손상, 매칭 실패, 라벨 이상, cross-ID pixel 중복 등. 제외 사유는 manifest와 이슈 CSV에 기록
