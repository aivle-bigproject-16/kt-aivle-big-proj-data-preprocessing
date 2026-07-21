# 🔋 배터리 결함 이미지 데이터 전처리 파이프라인 (v4.0)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-12.2.0-2C2D72)
![Shapely](https://img.shields.io/badge/Shapely-2.1.2-3AAFA9)
![YOLO](https://img.shields.io/badge/YOLOv11-det%20%2B%20seg-00FFFF?logo=ultralytics&logoColor=black)
![Tests](https://img.shields.io/badge/tests-45_passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**KT AIVLE School 9기 빅프로젝트 16조**의 CT·RGB 배터리 결함 이미지 전처리 코드입니다. 원본을 읽기 전용으로 스캔·검증하고, Battery ID 단위로 분할해 YOLO detection·segmentation 학습 데이터와 감사 보고서를 결정론적으로 생성합니다.

## v4.0 핵심 변경

CT polygon 복구·ROI clipping·multipart 분리 후 각 유효 조각의 정규화 좌표를 소수점 8자리로 직렬화합니다. 이 좌표에서 계산한 개별 bbox 면적비의 이미지별 최댓값을 `porosity_bbox_max_ratio`로 정의합니다.

```text
porosity_bbox_max_ratio >= 0.25
```

위 조건을 만족하는 CT 이미지는 Battery ID 통계, Test/development 선정, 5-fold 층화 전에 제외합니다. `porosity_area_sum_ratio`는 제외 기준이 아니라 EDA와 fold 층화에만 사용합니다. 정확히 0.25인 경계도 제외합니다.

```text
scan·검증
→ polygon 복구·ROI clipping·multipart 분리
→ porosity_bbox_max_ratio 계산
→ bbox 비율 25% 이상 CT 이미지 선제 제외
→ 제외 후 Battery ID 통계 재계산
→ CT Test 7 ID / development 40 ID 선정
→ development 5-fold 층화
→ dry-run 보고서·승인
→ execute 재계산·검증·출력
```

## 품질 게이트와 승인

- 구조·누수 게이트는 예외 승인할 수 없습니다. 선택 CT에 bbox 비율 25% 이상 이미지가 남아도 즉시 실패합니다.
- §17.2 품질 경고는 `dryrun_warnings.csv`와 `quality_exceptions.csv`에 기록합니다. 진행하려면 대응 행에 `status=approved_exception`, `reviewer`, `reviewed_at`, `reason`을 모두 입력해야 합니다.
- §17.3 검토 경고는 `review_warnings.csv`에 기록하지만 자동으로 approve/execute를 차단하지 않습니다.
- `approval.json`은 raw fingerprint, seed, bbox 정책, 제외 후 ID 통계와 승인 산출물 SHA-256을 고정합니다. execute에서 모두 재검증합니다.

## 저장소 구조

```text
battery_v4_0_preprocessor/
├─ battery_v4_0/          # 스캔·geometry·선정·보고서·출력·승인 파이프라인
├─ tests/                 # 단위·통합 회귀 테스트
├─ pyproject.toml
├─ requirements.lock
└─ README.md              # 상세 설치·실행·보고서 설명

데이터_전처리_v4.0_전체구현계획.md
```

v3.6 코드는 이력 보존을 위해 기존 디렉터리에 유지합니다. v3.6 work-dir, 승인 파일, 선택 ID CSV는 v4.0에서 재사용할 수 없습니다.

## 실행

```powershell
cd battery_v4_0_preprocessor
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock

.\.venv\Scripts\python.exe -m battery_v4_0.cli dry-run `
  --raw-root "<raw 데이터 루트>" --work-dir "<v4.0 작업 폴더>" --seed 42 --jobs 8

.\.venv\Scripts\python.exe -m battery_v4_0.cli approve-selection `
  --work-dir "<v4.0 작업 폴더>" --approved-by "<검토자>" --seed 42

.\.venv\Scripts\python.exe -m battery_v4_0.cli execute `
  --raw-root "<raw 데이터 루트>" --work-dir "<v4.0 작업 폴더>" `
  --output "<v4.0 출력 폴더>" --seed 42 --jobs 8
```

상세한 보고서, 품질 예외 승인, `training-view` 사용법은 [`battery_v4_0_preprocessor/README.md`](battery_v4_0_preprocessor/README.md)를 참고하세요. 확정 정책은 [`데이터_전처리_v4.0_전체구현계획.md`](데이터_전처리_v4.0_전체구현계획.md)에 있습니다.

## 검증

```powershell
python -m compileall -q battery_v4_0 tests
python -m unittest discover -s tests -v
```

현재 제공된 전체 회귀 테스트 45건이 통과합니다.

## 라이선스

MIT License. 자세한 내용은 [`LICENSE`](LICENSE)를 참고하세요.
