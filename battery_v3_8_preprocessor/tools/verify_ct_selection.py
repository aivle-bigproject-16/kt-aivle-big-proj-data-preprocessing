"""CT 선정 로직을 dry-run 없이 검증한다.

dry-run 한 번이 약 37분이므로, 선정 로직을 조금 바꿀 때마다 전체를 다시 도는 것은
현실적이지 않다. 이 도구는 완료된 dry-run의 `manifest.csv`에서 CT 샘플을 복원한
뒤 파이프라인이 쓰는 함수(`_ct_id_contamination`, `ct_id_gate`, `_select_ct_test`,
`build_ct_folds`)를 그대로 호출해 결과를 즉시 보여준다. 약 90초면 끝난다.

`--check` 를 주면 산출 결과가 그 work-dir의 `selected_battery_ids_candidate.csv`와
일치하는지 확인한다. 코드를 고치기 전에 먼저 이 검사를 통과시켜야 한다. 통과해야만
이 도구가 파이프라인을 대변한다고 말할 수 있고, 그 뒤의 측정값을 신뢰할 수 있다.

v3.8 개발 중 이 절차가 없어 사고가 두 번 났다. 한 번은 별도 최적화기의 결과를
구현 결과로 착각해 계획서에 잘못된 수치를 적었고(§26.10), 한 번은 검증 코드가
`_swap_density` 를 직접 호출해 파이프라인 호출부의 설정을 타지 않았다(§26.12).

사용법:

    python tools/verify_ct_selection.py <work-dir>/reports
    python tools/verify_ct_selection.py <work-dir>/reports --check
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from battery_v3_8 import selection as sel
from battery_v3_8.metrics import sample_metrics, selected_samples as collect
from battery_v3_8.models import Sample

csv.field_size_limit(10_000_000)


def load_ct_samples(reports: Path) -> list[Sample]:
    """manifest.csv에서 CT 샘플을 복원한다.

    선정에 필요한 필드만 채운다. det_lines 는 개수만 의미가 있으므로 자리표시자
    문자열을 개수만큼 넣는다.
    """
    samples = []
    with (reports / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["modality"] != "CT":
                continue
            samples.append(Sample(
                sample_id=row["sample_id"],
                modality="CT",
                battery_id=row["battery_id"],
                axis=row["axis"],
                application=row["application"],
                det_lines=["x"] * int(row["yolo_det_instance_count"]),
                seg_lines=["x"] * int(row["yolo_seg_instance_count"]),
                included_det=True,
                included_seg=True,
                porosity_area_sum_ratio=float(row["porosity_area_sum_ratio"]),
                porosity_bbox_max_ratio=float(row["porosity_bbox_max_ratio"]),
                porosity_polygon_count=int(row["porosity_polygon_count"]),
            ))
    return samples


def run_ct_selection(samples: list[Sample], seed: int = 42):
    for sample in samples:
        sample.pre_split_eligible = True
        sample.pre_split_exclusion_reason = ""
        sample.split_role = ""
        sample.fold_id = ""
        sample.selected = False
    contamination = sel._ct_id_contamination(samples)
    population = sel.apply_pre_split_policy(samples)
    ct_ids = [stats for stats in sel._id_stats(population) if stats.modality == "CT"]
    ct_ids, gate_rows = sel.ct_id_gate(ct_ids, contamination)
    test, development = sel._select_ct_test(ct_ids, seed, None)
    sel._select_ct_development_samples(development)
    for stats in test:
        stats.selected_samples = list(stats.samples)
        stats.split_role = "test"
    # 파이프라인과 같은 헬퍼를 호출한다. 여기서 갈라지면 검증이 다른 것을 재게 된다.
    groups, target = sel.build_ct_folds(development, seed)
    return gate_rows, test, groups, target


def report(gate_rows, test, groups, target) -> float:
    development = [stats for members in groups.values() for stats in members]
    dev_metrics = sample_metrics(collect(development))
    test_metrics = sample_metrics(sample for stats in test for sample in stats.samples)

    print(f"ID 게이트 제외 {len(gate_rows)}건: {[str(r['battery_id']) for r in gate_rows]}")
    print(f"Test {len(test)}개: {sorted(stats.battery_id for stats in test)}")
    worst = defect_spread = image_share = 0.0
    for name in sorted(groups):
        metrics = sample_metrics(collect(groups[name], name, None))
        deviation = metrics.annotations_per_image / target - 1
        worst = max(worst, abs(deviation))
        defect_spread = max(defect_spread, abs(metrics.defect_image_ratio - dev_metrics.defect_image_ratio))
        image_share = max(image_share, abs(metrics.images / dev_metrics.images - 0.2))
        print(f"  fold {name}: {metrics.images:>6,}장 apd={metrics.annotations_per_image:.6f} "
              f"({deviation:+.2%}) defect={metrics.defect_image_ratio:.4f} "
              f":: {sorted(stats.battery_id for stats in groups[name])}")

    gate = "PASS" if worst <= 0.20 else "FAIL"
    print(f"\nfold 최악 밀도 편차 : {worst:+.2%}  {gate} (여유 {20 - worst * 100:.2f}%p)")
    print(f"Test/dev 밀도비     : {test_metrics.annotations_per_image / dev_metrics.annotations_per_image:.3f}배")
    print(f"Test/dev defect비   : {test_metrics.defect_image_ratio / dev_metrics.defect_image_ratio:.3f}배")
    print(f"fold defect 이탈    : {defect_spread:.4f}")
    print(f"fold 이미지 비율 이탈: {image_share:.4f}")
    return worst


def check_against_recorded(reports: Path, test, groups) -> bool:
    expected_test: set[str] = set()
    expected_folds: dict[str, str] = {}
    with (reports / "selected_battery_ids_candidate.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["modality"] != "CT":
                continue
            if row["split_role"] == "test":
                expected_test.add(row["battery_id"])
            else:
                expected_folds[row["battery_id"]] = row["fold_id"]

    got_test = {stats.battery_id for stats in test}
    got_folds = {stats.battery_id: name for name, members in groups.items() for stats in members}
    test_ok, folds_ok = got_test == expected_test, got_folds == expected_folds
    print(f"\n기록된 dry-run과 대조: Test {'일치' if test_ok else '불일치'}, "
          f"fold {'일치' if folds_ok else '불일치'}")
    if not test_ok:
        print(f"  기록 {sorted(expected_test)}")
        print(f"  산출 {sorted(got_test)}")
    if not folds_ok:
        differing = {
            key: (expected_folds.get(key), got_folds.get(key))
            for key in set(expected_folds) | set(got_folds)
            if expected_folds.get(key) != got_folds.get(key)
        }
        print(f"  차이 {differing}")
    return test_ok and folds_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="dry-run 없이 CT 선정 로직을 검증한다.")
    parser.add_argument("reports", type=Path, help="완료된 dry-run의 reports 폴더")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="기록된 dry-run 결과와 일치해야 성공으로 종료한다.")
    args = parser.parse_args()

    samples = load_ct_samples(args.reports)
    print(f"CT 샘플 {len(samples):,}장, ID {len({s.battery_id for s in samples})}개 복원\n")
    gate_rows, test, groups, target = run_ct_selection(samples, args.seed)
    report(gate_rows, test, groups, target)

    if args.check:
        return 0 if check_against_recorded(args.reports, test, groups) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
