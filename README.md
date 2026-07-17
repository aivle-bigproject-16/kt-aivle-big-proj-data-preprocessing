# 🔋 배터리 결함 이미지 데이터 전처리 파이프라인 (v3.6)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-12.2.0-2C2D72)
![Shapely](https://img.shields.io/badge/Shapely-2.1.2-3AAFA9)
![YOLO](https://img.shields.io/badge/YOLOv11-det%20%2B%20seg-00FFFF?logo=ultralytics&logoColor=black)
![Tests](https://img.shields.io/badge/tests-32_passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**KT AIVLE School 9기 빅프로젝트 16조** — 이차전지 셀 결함 검사 플랫폼의 데이터 전처리 파이프라인입니다. AIHub 배터리 불량 이미지(CT·RGB) 원본을 스캔·검증하여 **YOLOv11 detection·segmentation 학습 데이터**, 원본 JSON 보존본, manifest, 품질 보고서를 결정론적으로 생성합니다.

## ✨ 핵심 기능

- 이미지와 JSON을 **원본 구역·파일 stem 기준 1:1 매칭**하고, 파일명·배터리 ID·영상 크기·CT ROI·JSON 구조를 검증합니다.
- 원본 JSON은 **byte 단위로 보존(SHA-256 대조)** 하고, YOLO detection/segmentation 라벨을 파생 생성합니다. Shapely로 폴리곤 복구·clipping·multipart 분리를 처리합니다.
- 디코딩된 RGB 픽셀 해시로 **동일 ID 내부 중복**과 **cross-ID 중복**을 제거하고, 모든 split/fold 쌍에 대해 **battery ID·pixel hash 누수를 0으로 감사**합니다.
- 배터리 ID 기준으로 분할해 누수를 차단합니다 — **CT: Test 7 + development 40 ID로 5-fold CV**, **RGB: 대표 160 ID를 128/16/16**로 분할합니다.
- `defect_image_ratio`(1차)와 `annotations_per_image`(2차, ±20% 게이트)로 fold·split 균형을 맞추고, CT는 **porosity 면적합 층화**를 추가로 반영합니다.
- **dry-run → approve → execute** 3단계 승인 절차로, 검토·승인 후에만 실데이터를 생성하며 staging 역검증을 통과해야 최종 output으로 원자적 승격합니다.
- 동일 raw·seed·환경에서 **멤버십·라벨·manifest·보고서가 완전히 재현**되는 결정론을 보장합니다.

## 📊 품질 게이트

품질 게이트는 두 계층으로 분리됩니다.

- **하드 게이트(§17.2)** — 밀도 ±20% 초과, 필수 클래스 부재, det/seg 제외율 초과 등. `dryrun_warnings.csv`에 기록되며 approve·execute를 즉시 중단합니다.
- **검토 경고(§17.3)** — 면적 구간 쏠림, fold 이미지 비율, Test 이미지 부족 등. `review_warnings.csv`로 라우팅되어 검토자가 acknowledge한 뒤에만 진행합니다. 경고를 숨기거나 누수 게이트를 완화하지 않습니다.

대용량 라벨 유지 결정과 근본 원인은 `데이터_전처리_v3.6_전체구현계획.md` §23·§24에 기록되어 있습니다.

## 📂 저장소 구조

```
battery_v3_6_preprocessor/
├─ battery_v3_6/
│  ├─ cli.py             # dry-run / approve / execute 진입점
│  ├─ scan.py            # 전체 탐색·검증·중복 제거
│  ├─ parsing.py         # 파일명·JSON·클래스 파싱
│  ├─ geometry.py        # 폴리곤 복구·YOLO 변환·면적
│  ├─ ct_area.py         # CT 면적 구간·층화 feature
│  ├─ metrics.py         # 공통 지표
│  ├─ selection.py       # CT/RGB ID 선정·배정·swap
│  ├─ reports.py         # manifest·CSV·EDA
│  ├─ output.py          # 이미지·JSON·TXT·ZIP 출력
│  ├─ pipeline.py        # 승인·staging·execute
│  ├─ deterministic.py   # stable hash·quota·sampling
│  ├─ models.py          # Sample·IdStats 등
│  └─ training_view.py   # Ultralytics 학습 뷰
├─ tests/                # 단위·통합 테스트 32건
├─ run_dryrun_keepawake.ps1 / run_execute_keepawake.ps1  # 장시간 실행 래퍼(절전 방지)
├─ prepare_training_view.py
├─ pyproject.toml · requirements.lock
├─ README.md             # 설치·실행 상세
└─ V3.6_계획_코드_충돌검사.md
데이터_전처리_v3.6_전체구현계획.md   # 독립 확정 계획서(이 문서만으로 재구현 가능)
폴더_구성.md
```

## 🚀 실행

```powershell
cd battery_v3_6_preprocessor
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock

# 1) dry-run — 스캔·검증·선정·보고서만, 출력 생성 안 함
.\.venv\Scripts\python.exe -m battery_v3_6.cli dry-run `
  --raw-root "<raw 데이터 루트>" --work-dir "<작업 폴더>" --seed 42 --jobs 8

# 2) approve — dry-run 보고서 검토 후 Test ID·선정 확정
.\.venv\Scripts\python.exe -m battery_v3_6.cli approve-selection `
  --work-dir "<작업 폴더>" --approved-by "<검토자>"

# 3) execute — staging 생성·역검증 후 최종 output으로 승격
.\.venv\Scripts\python.exe -m battery_v3_6.cli execute `
  --raw-root "<raw 데이터 루트>" --work-dir "<작업 폴더>" --output "<출력 폴더>" --seed 42 --jobs 8
```

상세 실행법은 `battery_v3_6_preprocessor/README.md`를 참고합니다.

## ⚠️ 데이터·산출물

원본 데이터, 작업 폴더(`*_work`), 출력 폴더(`*_output`)는 저장소에 포함하지 않습니다. `raw-root`는 AIHub 배터리 불량 이미지 데이터의 `3.개방데이터\1.데이터` 위치를 지정합니다.

## 📄 라이선스

MIT License. 자세한 내용은 [`LICENSE`](LICENSE)를 참고합니다.
