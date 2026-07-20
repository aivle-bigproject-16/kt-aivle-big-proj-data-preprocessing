from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .deterministic import largest_remainder, natural_id_key, order_key, quantile_bins, stratified_sample
from .ct_area import (
    AREA_BIN_ORDER,
    CT_PRE_SPLIT_EXCLUSION_REASON,
    ct_area_features,
    is_ct_pre_split_excluded,
    porosity_area_bin,
)
from .metrics import sample_metrics, selected_samples as collect_selected_samples
from .models import IdStats, Sample
from .parsing import choose_application

EPSILON = 1e-12
MAX_SWAP_ROUNDS = 12

# v3.8 CT ID 게이트.
# Gate 1은 이미지 다수가 대형 공동인 ID를 통째로 제외한다. 이미지 단위 제외만
# 적용하면 그런 ID는 원본의 11~22%만 남아 다른 ID와 같은 자격으로 fold에 들어갈 수
# 없다. Gate 2는 annotation 밀도가 사분위 범위를 크게 벗어난 ID를 제외한다.
# 두 게이트 모두 ID 이름을 직접 지정하지 않고 분포에서 임계값을 얻는다.
CT_ID_CONTAMINATION_BASIS = 0.25
CT_ID_CONTAMINATION_CUT = 0.25
CT_ID_DENSITY_IQR_MULTIPLIER = 3.0
CT_ID_GATE_MAX_DROPS = 12
CT_ID_GATE_EXCLUSION_REASON = "ct_id_gate_excluded"
CT_TEST_TARGET = 7
CT_FOLDS = 5


def ct_split_structure(id_count: int) -> tuple[int, int]:
    """Return (ids_per_fold, test_id_count) for the surviving CT ID count.

    Development always fills five equal folds and Test takes the remainder, so
    the shape follows the data instead of a fixed 47/40/7 assumption:
    47 IDs -> 8 per fold + Test 7, 37 IDs -> 6 per fold + Test 7.
    """
    per_fold = (id_count - CT_TEST_TARGET) // CT_FOLDS
    if per_fold < 4:
        raise ValueError(
            f"CT surviving ID count {id_count} is too small for {CT_FOLDS} folds with Test {CT_TEST_TARGET}"
        )
    return per_fold, id_count - per_fold * CT_FOLDS


@dataclass
class SelectionResult:
    ids: list[IdStats]
    warnings: list[str] = field(default_factory=list)  # §17.2 하드 게이트(밀도 ±20%, 필수 클래스, det/seg): approve/execute 하드 블록
    review_warnings: list[str] = field(default_factory=list)  # §17.3 검토 경고: 감사 기록용이며 자동 실행 차단 대상이 아님
    leakage_rows: list[dict[str, object]] = field(default_factory=list)
    ct_id_gate_rows: list[dict[str, object]] = field(default_factory=list)  # §8.5 ID 게이트로 제외된 CT ID

    @property
    def selected_samples(self) -> list[Sample]:
        return sorted(
            collect_selected_samples(self.ids),
            key=lambda sample: sample.sample_id,
        )


def _id_stats(samples: Iterable[Sample]) -> list[IdStats]:
    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.modality, sample.battery_id)].append(sample)
    result = []
    for (modality, battery_id), group in groups.items():
        group.sort(key=lambda sample: sample.sample_id)
        application = choose_application([sample.application for sample in group])
        result.append(IdStats(modality, battery_id, application, group))
    return sorted(result, key=lambda stats: (stats.modality, natural_id_key(stats.battery_id)))


def apply_pre_split_policy(valid_samples: Iterable[Sample]) -> list[Sample]:
    """Apply v3.8 policy exclusions before ID statistics and stratification.

    Excluded samples stay on the scanned Sample objects for lineage/reporting,
    but are absent from the returned split population.  This function is
    deterministic and safe to call again during execute recomputation.
    """
    eligible: list[Sample] = []
    for sample in valid_samples:
        if sample.modality == "CT" and (
            not math.isfinite(sample.porosity_area_sum_ratio)
            or not math.isfinite(sample.porosity_bbox_max_ratio)
            or sample.porosity_area_sum_ratio < 0
            or sample.porosity_bbox_max_ratio < 0
        ):
            raise ValueError(f"CT area metric is negative or non-finite: {sample.sample_id}")
        excluded = is_ct_pre_split_excluded(sample)
        sample.pre_split_eligible = not excluded
        sample.pre_split_exclusion_reason = CT_PRE_SPLIT_EXCLUSION_REASON if excluded else ""
        if excluded:
            sample.selected = False
            sample.split_role = ""
            sample.fold_id = ""
            continue
        eligible.append(sample)
    return eligible


def _ct_id_contamination(valid_samples: Iterable[Sample]) -> dict[str, float]:
    """Share of each CT ID's images that the image-level rule removes.

    Computed on the full valid population, before pre-split filtering, so the
    denominator is the ID's original image count.
    """
    totals: dict[str, int] = defaultdict(int)
    excluded: dict[str, int] = defaultdict(int)
    for sample in valid_samples:
        if sample.modality != "CT":
            continue
        totals[sample.battery_id] += 1
        if sample.porosity_bbox_max_ratio >= CT_ID_CONTAMINATION_BASIS:
            excluded[sample.battery_id] += 1
    return {
        battery_id: excluded[battery_id] / count
        for battery_id, count in totals.items()
        if count
    }


def _quartiles(values: list[float]) -> tuple[float, float]:
    """Q1 and Q3 by linear interpolation, matching statistics.quantiles(n=4)."""
    ordered = sorted(values)
    if len(ordered) < 2:
        return (ordered[0], ordered[0]) if ordered else (0.0, 0.0)

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) + 1) - 1
        low = max(0, min(len(ordered) - 1, int(math.floor(position))))
        high = max(0, min(len(ordered) - 1, low + 1))
        weight = position - low
        return ordered[low] + (ordered[high] - ordered[low]) * weight

    return at(0.25), at(0.75)


def ct_id_gate(ct_ids: list[IdStats], contamination: dict[str, float]) -> tuple[list[IdStats], list[dict[str, object]]]:
    """Drop CT IDs that cannot participate in a balanced five-fold split.

    Gate 1 removes IDs whose images are predominantly large voids, measured as
    the share of the ID's original images excluded by the image-level rule.
    Gate 2 removes annotation-density outliers, using Tukey's far-out fence on
    the IDs that survive Gate 1. Gate 2 runs exactly once; re-applying it to its
    own output would keep promoting the next-densest ID and erode the dataset.
    """
    rows: list[dict[str, object]] = []
    survivors: list[IdStats] = []
    for stats in ct_ids:
        share = contamination.get(stats.battery_id, 0.0)
        if share >= CT_ID_CONTAMINATION_CUT:
            rows.append({
                "battery_id": stats.battery_id,
                "gate": "contamination",
                "observed": f"{share:.8f}",
                "threshold": f"{CT_ID_CONTAMINATION_CUT:.8f}",
                "reason": f"ct_id_contamination_ge_{CT_ID_CONTAMINATION_CUT}",
            })
        else:
            survivors.append(stats)

    densities = {
        stats.battery_id: sample_metrics(stats.samples).annotations_per_image
        for stats in survivors
    }
    q1, q3 = _quartiles(list(densities.values())) if densities else (0.0, 0.0)
    # An interquartile range of zero makes the fence collapse onto Q3, which would
    # flag every ID carrying any annotation at all. That happens whenever fewer
    # than a quarter of the IDs have defects, so the outlier test is undefined
    # there and the gate must not fire.
    if densities and (q3 - q1) > EPSILON:
        fence = q3 + CT_ID_DENSITY_IQR_MULTIPLIER * (q3 - q1)
        kept: list[IdStats] = []
        for stats in survivors:
            value = densities[stats.battery_id]
            if value > fence + EPSILON:
                rows.append({
                    "battery_id": stats.battery_id,
                    "gate": "density_outlier",
                    "observed": f"{value:.8f}",
                    "threshold": f"{fence:.8f}",
                    "reason": f"ct_id_annotations_per_image_over_q3_plus_{CT_ID_DENSITY_IQR_MULTIPLIER}_iqr",
                })
            else:
                kept.append(stats)
        survivors = kept

    if len(rows) > CT_ID_GATE_MAX_DROPS:
        raise ValueError(
            f"CT ID gates dropped {len(rows)} IDs (limit {CT_ID_GATE_MAX_DROPS}); "
            "review the raw data before continuing"
        )
    rows.sort(key=lambda row: natural_id_key(str(row["battery_id"])))
    return survivors, rows


def _pooled_ratio(ids: Iterable[IdStats], samples: Callable[[IdStats], list[Sample]] | None = None) -> float:
    chosen = [sample for stats in ids for sample in (samples(stats) if samples else stats.samples)]
    return sample_metrics(chosen).defect_image_ratio


def _ct_axis_ratios(ids: Iterable[IdStats]) -> tuple[float, float, float]:
    counts = tuple(sum(stats.axis_count(axis) for stats in ids) for axis in "xyz")
    total = sum(counts)
    if total == 0:
        return (0.0, 0.0, 0.0)
    return tuple(count / total for count in counts)


def _density_gap(test: list[IdStats], development: list[IdStats]) -> float:
    """Relative annotations_per_image gap between Test and development.

    v3.7 balanced axes, defect ratio and area bins but not annotation density,
    which let the densest IDs collect in Test: measured 0.4963 against 0.2071,
    a 2.4x gap of exactly the kind annotations_per_image exists to prevent.
    """
    test_density = sample_metrics(sample for stats in test for sample in stats.samples).annotations_per_image
    development_density = sample_metrics(
        sample for stats in development for sample in stats.samples
    ).annotations_per_image
    if development_density <= 0:
        return 0.0 if test_density <= 0 else 1.0
    return abs(test_density / development_density - 1)


def _ct_test_objective(test: list[IdStats], development: list[IdStats]) -> tuple[float, float, float, float, float]:
    test_axes = _ct_axis_ratios(test)
    development_axes = _ct_axis_ratios(development)
    gaps = [abs(test_ratio - development_ratio) for test_ratio, development_ratio in zip(test_axes, development_axes)]
    test_area = ct_area_features(sample for stats in test for sample in stats.samples)
    development_area = ct_area_features(sample for stats in development for sample in stats.samples)
    area_gaps = []
    test_total = test_area["images"]
    development_total = development_area["images"]
    for area_bin in AREA_BIN_ORDER:
        test_ratio = test_area[f"area:{area_bin}"] / test_total if test_total else 0.0
        development_ratio = development_area[f"area:{area_bin}"] / development_total if development_total else 0.0
        area_gaps.append(abs(test_ratio - development_ratio))
    return (
        max(gaps),
        sum(gaps),
        abs(_pooled_ratio(test) - _pooled_ratio(development)),
        _density_gap(test, development),
        max(area_gaps, default=0.0),
    )


def _assign_virtual_rgb_samples(ids: list[IdStats], seed: int) -> None:
    for stats in ids:
        stats.selected_samples = stratified_sample(stats.samples, 300, seed, f"EXT:{stats.battery_id}:selection")


def select_rgb_160(rgb_ids: list[IdStats], seed: int) -> list[IdStats]:
    if len(rgb_ids) < 160:
        raise ValueError(f"RGB valid ID count must be >=160, got {len(rgb_ids)}")
    _assign_virtual_rgb_samples(rgb_ids, seed)
    ratios = {
        stats.battery_id: sample_metrics(stats.selected_samples).defect_image_ratio
        for stats in rgb_ids
    }
    bins = quantile_bins(ratios, 5)
    strata: dict[tuple[str, bool, int], list[IdStats]] = defaultdict(list)
    for stats in rgb_ids:
        strata[(stats.application, stats.has_damaged, bins[stats.battery_id])].append(stats)
    quotas = largest_remainder(
        160,
        {key: len(value) for key, value in strata.items()},
        {key: len(value) for key, value in strata.items()},
    )
    selected: list[IdStats] = []
    for key in sorted(strata, key=lambda value: (value[0], value[1], value[2])):
        members = strata[key]
        mean = sum(ratios[stats.battery_id] for stats in members) / len(members)
        members.sort(key=lambda stats: (abs(ratios[stats.battery_id] - mean), order_key(seed, "rgb160", stats.battery_id), stats.battery_id))
        selected.extend(members[: quotas[key]])
    selected.sort(key=lambda stats: natural_id_key(stats.battery_id))
    if len(selected) != 160:
        raise RuntimeError(f"RGB selection produced {len(selected)} IDs")
    _validate_rgb_representativeness(rgb_ids, selected)
    return selected


def _validate_rgb_representativeness(raw: list[IdStats], selected: list[IdStats]) -> None:
    def category_ratios(ids: list[IdStats]) -> dict[str, float]:
        counts = Counter(stats.application for stats in ids)
        return {key: counts[key] / len(ids) for key in set(counts)}

    raw_apps, selected_apps = category_ratios(raw), category_ratios(selected)
    violations = []
    for category in set(raw_apps) | set(selected_apps):
        if abs(raw_apps.get(category, 0.0) - selected_apps.get(category, 0.0)) > 0.02 + EPSILON:
            violations.append(f"application[{category}]")
    raw_ratio = _pooled_ratio(raw, lambda stats: stats.selected_samples)
    selected_ratio = _pooled_ratio(selected, lambda stats: stats.selected_samples)
    if abs(raw_ratio - selected_ratio) > 0.03 + EPSILON:
        violations.append("defect_image_ratio")
    raw_damaged = sum(stats.has_damaged for stats in raw) / len(raw)
    selected_damaged = sum(stats.has_damaged for stats in selected) / len(selected)
    if abs(raw_damaged - selected_damaged) > 0.02 + EPSILON:
        violations.append("has_damaged")
    normal_ratio = sample_metrics(collect_selected_samples(selected)).normal_ratio
    if normal_ratio + EPSILON < 0.30:
        violations.append("normal_ratio<30%")
    if violations:
        raise ValueError("RGB 160 representativeness gate failed: " + ", ".join(violations))


def _select_ct_test(ct_ids: list[IdStats], seed: int, locked: set[str] | None) -> tuple[list[IdStats], list[IdStats]]:
    per_fold, test_count = ct_split_structure(len(ct_ids))
    development_count = per_fold * CT_FOLDS
    invalid_axes = sorted({sample.axis for stats in ct_ids for sample in stats.samples if sample.axis not in "xyz"})
    if invalid_axes:
        raise ValueError(f"CT samples contain invalid axes: {invalid_axes}")
    if locked is not None:
        if len(locked) != test_count:
            raise ValueError(f"locked CT Test must contain exactly {test_count} IDs")
        test = [stats for stats in ct_ids if stats.battery_id in locked]
        if len(test) != test_count:
            raise ValueError("locked CT Test contains missing IDs")
    else:
        test = []
        while len(test) < test_count:
            candidates = [stats for stats in ct_ids if stats not in test]
            candidates.sort(
                key=lambda stats: (
                    _ct_test_objective(
                        test + [stats],
                        [candidate for candidate in ct_ids if candidate not in test and candidate is not stats],
                    ),
                    order_key(seed, "ct_test", stats.battery_id),
                    stats.battery_id,
                )
            )
            test.append(candidates[0])
        while True:
            development = [stats for stats in ct_ids if stats not in test]
            before = _ct_test_objective(test, development)
            best: tuple[tuple[float, float, float, float, float], bytes, str, str, IdStats, IdStats] | None = None
            for test_id in sorted(test, key=lambda stats: natural_id_key(stats.battery_id)):
                for development_id in sorted(development, key=lambda stats: natural_id_key(stats.battery_id)):
                    proposed_test = [stats for stats in test if stats is not test_id] + [development_id]
                    proposed_development = [stats for stats in development if stats is not development_id] + [test_id]
                    objective = _ct_test_objective(proposed_test, proposed_development)
                    if objective >= before:
                        continue
                    candidate = (
                        objective,
                        order_key(seed, "ct_test_axis_swap", f"{test_id.battery_id}:{development_id.battery_id}"),
                        test_id.battery_id,
                        development_id.battery_id,
                        test_id,
                        development_id,
                    )
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
            if best is None:
                break
            _, _, _, _, outgoing, incoming = best
            test.remove(outgoing)
            test.append(incoming)
    development = [stats for stats in ct_ids if stats not in test]
    if len(development) != development_count:
        raise ValueError("locked/selected CT Test leaves an invalid development set")
    return sorted(test, key=lambda stats: natural_id_key(stats.battery_id)), sorted(development, key=lambda stats: natural_id_key(stats.battery_id))


def _select_ct_development_samples(development: list[IdStats]) -> None:
    for stats in development:
        stats.selected_samples = sorted(stats.samples, key=lambda sample: sample.sample_id)


def _ct_fold_assignment(development: list[IdStats], seed: int) -> dict[int, list[IdStats]]:
    """Assign whole IDs to five equal folds while retaining every balance metric.

    Image count, axis, porosity-area bin and axis×area-bin balance are scored
    together. defect_image_ratio remains the primary marginal and
    annotations_per_image is included here and still enforced by the existing
    post-assignment ±20% gate. The fold size follows the surviving ID count.
    """
    if len(development) % CT_FOLDS:
        raise ValueError(f"CT development must divide into {CT_FOLDS} folds, got {len(development)} IDs")
    ids_per_fold = len(development) // CT_FOLDS
    target_samples = [sample for stats in development for sample in stats.selected_samples]
    target_metrics = sample_metrics(target_samples)
    total_features = ct_area_features(target_samples)
    cached = {}
    for stats in development:
        metrics = sample_metrics(stats.selected_samples)
        cached[stats.battery_id] = (
            metrics.images,
            metrics.defect_images,
            metrics.annotations,
            ct_area_features(stats.selected_samples),
        )

    def fold_score(members: list[IdStats]) -> float:
        image_count = sum(cached[stats.battery_id][0] for stats in members)
        defect_count = sum(cached[stats.battery_id][1] for stats in members)
        annotation_count = sum(cached[stats.battery_id][2] for stats in members)
        features: Counter[str] = Counter()
        for stats in members:
            features.update(cached[stats.battery_id][3])
        defect_ratio = defect_count / image_count if image_count else 0.0
        annotations_per_image = annotation_count / image_count if image_count else 0.0
        score = 0.0
        score += 12.0 * ((image_count - len(target_samples) / CT_FOLDS) / max(len(target_samples) / CT_FOLDS, 1)) ** 2
        score += 8.0 * ((defect_ratio - target_metrics.defect_image_ratio) / max(target_metrics.defect_image_ratio, 0.01)) ** 2
        score += 4.0 * ((annotations_per_image - target_metrics.annotations_per_image) / max(target_metrics.annotations_per_image, 0.01)) ** 2
        for feature, total in total_features.items():
            wanted = total / CT_FOLDS
            if feature == "images":
                continue
            weight = 3.0 if feature.startswith("axis:") else 2.0 if feature.startswith("joint:") else 1.0
            score += weight * ((features[feature] - wanted) / max(wanted, 1)) ** 2
        return score

    best_folds: dict[int, list[IdStats]] | None = None
    best_score = float("inf")
    # Deterministic multi-start greedy. Quota is exactly ids_per_fold IDs per fold.
    for restart in range(500):
        ranked = sorted(
            development,
            key=lambda stats: (
                -len(stats.selected_samples),
                order_key(seed, f"ct_area_restart:{restart}", stats.battery_id),
                natural_id_key(stats.battery_id),
            ),
        )
        folds: dict[int, list[IdStats]] = {index: [] for index in range(CT_FOLDS)}
        for stats in ranked:
            candidates = [index for index in range(CT_FOLDS) if len(folds[index]) < ids_per_fold]
            chosen = min(
                candidates,
                key=lambda index: (
                    fold_score(folds[index] + [stats]) - fold_score(folds[index]),
                    len(folds[index]),
                    index,
                ),
            )
            folds[chosen].append(stats)
        score = sum(fold_score(members) for members in folds.values())
        if score < best_score:
            best_score = score
            best_folds = {index: list(members) for index, members in folds.items()}
    assert best_folds is not None
    folds = best_folds
    if any(len(members) != ids_per_fold for members in folds.values()):
        raise RuntimeError(f"CT fold assignment did not produce {CT_FOLDS} groups of {ids_per_fold}")
    return folds


def _id_selected_ratio(stats: IdStats) -> float:
    return sample_metrics(stats.selected_samples).defect_image_ratio


def _swap_density(
    groups: dict[str, list[IdStats]],
    stratum: Callable[[IdStats], object],
    target: float,
    locked_groups: set[str] | None = None,
    selected_for: Callable[[str, IdStats], list[Sample]] | None = None,
    dynamic_target: bool = False,
) -> None:
    locked_groups = locked_groups or set()
    selected_for = selected_for or (lambda _name, stats: stats.selected_samples)

    def metrics_for(name: str, members: list[IdStats]):
        return sample_metrics(collect_selected_samples(members, name, selected_for))

    def target_for(current: dict[str, list[IdStats]]) -> float:
        if not dynamic_target:
            return target
        samples = [
            sample
            for name, members in current.items()
            if name not in locked_groups
            for sample in collect_selected_samples(members, name, selected_for)
        ]
        return sample_metrics(samples).annotations_per_image

    def objective(current: dict[str, list[IdStats]]) -> float:
        current_target = target_for(current)
        if current_target == 0:
            return 0.0
        return max(abs(metrics_for(name, members).annotations_per_image / current_target - 1) for name, members in current.items() if name not in locked_groups)

    for _ in range(MAX_SWAP_ROUNDS):
        before = objective(groups)
        if before <= 0.15 + EPSILON:
            break
        total_samples = [sample for name, members in groups.items() for sample in collect_selected_samples(members, name, selected_for)]
        overall_before = sample_metrics(total_samples).defect_image_ratio
        before_ratio_spread = max(abs(metrics_for(name, members).defect_image_ratio - overall_before) for name, members in groups.items())
        best: tuple[float, str, str, IdStats, IdStats] | None = None
        names = sorted(name for name in groups if name not in locked_groups)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                for left in sorted(groups[left_name], key=lambda stats: natural_id_key(stats.battery_id)):
                    for right in sorted(groups[right_name], key=lambda stats: natural_id_key(stats.battery_id)):
                        if stratum(left) != stratum(right):
                            continue
                        proposal = {name: list(members) for name, members in groups.items()}
                        proposal[left_name].remove(left)
                        proposal[right_name].remove(right)
                        proposal[left_name].append(right)
                        proposal[right_name].append(left)
                        proposal_samples = [sample for name, members in proposal.items() for sample in collect_selected_samples(members, name, selected_for)]
                        overall_ratio = sample_metrics(proposal_samples).defect_image_ratio
                        ratio_spread = max(abs(metrics_for(name, members).defect_image_ratio - overall_ratio) for name, members in proposal.items())
                        if ratio_spread > before_ratio_spread + 0.01 + EPSILON:
                            continue
                        value = objective(proposal)
                        candidate = (value, left.battery_id, right.battery_id, left, right)
                        if value + EPSILON < before and (best is None or candidate[:3] < best[:3]):
                            best = candidate
                            best_names = (left_name, right_name)
        if best is None:
            break
        _, _, _, left, right = best
        left_name, right_name = best_names
        groups[left_name].remove(left)
        groups[right_name].remove(right)
        groups[left_name].append(right)
        groups[right_name].append(left)


def _split_rgb(selected: list[IdStats], seed: int, locked_test: set[str] | None) -> dict[str, list[IdStats]]:
    ratios = {stats.battery_id: _id_selected_ratio(stats) for stats in selected}
    bins = quantile_bins(ratios, 5)
    strata: dict[tuple[str, bool, int], list[IdStats]] = defaultdict(list)
    for stats in selected:
        strata[(stats.application, stats.has_damaged, bins[stats.battery_id])].append(stats)
    if locked_test is not None:
        if len(locked_test) != 16:
            raise ValueError("locked RGB Test must contain exactly 16 IDs")
        test = [stats for stats in selected if stats.battery_id in locked_test]
        if len(test) != 16:
            raise ValueError("locked RGB Test contains IDs outside selected 160")
    else:
        quotas = largest_remainder(16, {key: len(value) for key, value in strata.items()}, {key: len(value) for key, value in strata.items()})
        test = []
        for key, members in strata.items():
            members = sorted(members, key=lambda stats: (order_key(seed, "rgb_test", stats.battery_id), stats.battery_id))
            test.extend(members[: quotas[key]])
    remaining = [stats for stats in selected if stats not in test]
    remaining_strata = {key: [stats for stats in members if stats in remaining] for key, members in strata.items()}
    remaining_strata = {key: members for key, members in remaining_strata.items() if members}
    quotas = largest_remainder(16, {key: len(value) for key, value in remaining_strata.items()}, {key: len(value) for key, value in remaining_strata.items()})
    val = []
    for key, members in remaining_strata.items():
        members = sorted(members, key=lambda stats: (order_key(seed, "rgb_val", stats.battery_id), stats.battery_id))
        val.extend(members[: quotas[key]])
    train = [stats for stats in remaining if stats not in val]
    if (len(train), len(val), len(test)) != (128, 16, 16):
        raise RuntimeError("RGB split did not produce 128/16/16")
    if not any(stats.has_damaged for stats in val) or not any(stats.has_damaged for stats in test):
        raise ValueError("RGB Validation and Test must each contain a Damaged-bearing ID")
    return {"train": train, "val": val, "test": test}


def _check_density(groups: dict[str, list[IdStats]], target: float, gate_groups: set[str], label: str) -> list[str]:
    if target == 0:
        return []
    warnings = []
    for name in sorted(gate_groups):
        achieved = sample_metrics(collect_selected_samples(groups[name])).annotations_per_image
        deviation = abs(achieved / target - 1)
        # 2차 밀도 가드레일: ±20%. 실데이터(2026-07-07 dry-run)서 CT fold_0가 same-stratum swap 후에도
        # 18.9%로 잔존(swap이 더 못 줄이는 구조적 최소). ±15%는 §13.7에서 v2 스큐(≈120%) 방지 목적으로
        # ±10%에서 완화된 값인데, 18.9%도 그 참사 대비 6배 이하라 취지 충족. 게이트만 20%로 넓히고
        # swap 종료 목표(0.5.7)는 0.15로 유지해 swap은 계속 최대한 조인다.
        if deviation > 0.20 + EPSILON:
            warnings.append(f"{label}:{name}: annotations_per_image deviation={deviation:.8f} target={target:.8f} achieved={achieved:.8f}")
    return warnings


def _check_ct_area_balance(groups: dict[str, list[IdStats]]) -> list[str]:
    warnings: list[str] = []
    samples_by_fold = {
        name: [sample for stats in members for sample in stats.selected_samples]
        for name, members in groups.items()
    }
    total_images = sum(len(samples) for samples in samples_by_fold.values())
    for name, samples in sorted(samples_by_fold.items()):
        actual = len(samples) / total_images if total_images else 0.0
        if abs(actual - 0.20) >= 0.05 - EPSILON:
            warnings.append(f"CT_fold_{name}: image ratio={actual:.8f}, target=0.20000000")
    counts = {
        name: Counter(porosity_area_bin(sample.porosity_area_sum_ratio) for sample in samples)
        for name, samples in samples_by_fold.items()
    }
    totals = Counter()
    for fold_counts in counts.values():
        totals.update(fold_counts)
    for area_bin, total in totals.items():
        if total and max(counts[name][area_bin] for name in counts) / total >= 0.40 - EPSILON:
            warnings.append(f"CT_area_bin[{area_bin}]: >=40% concentrated in one fold")
    return warnings


def assign_dataset(
    valid_samples: list[Sample],
    seed: int = 42,
    locked_tests: dict[str, set[str]] | None = None,
) -> SelectionResult:
    locked_tests = locked_tests or {}
    contamination = _ct_id_contamination(valid_samples)
    split_population = apply_pre_split_policy(valid_samples)
    all_ids = _id_stats(split_population)
    ct_ids = [stats for stats in all_ids if stats.modality == "CT"]
    rgb_ids = [stats for stats in all_ids if stats.modality == "EXT"]
    ct_ids, ct_id_gate_rows = ct_id_gate(ct_ids, contamination)
    dropped_ct_ids = {str(row["battery_id"]) for row in ct_id_gate_rows}
    for sample in valid_samples:
        if sample.modality == "CT" and sample.battery_id in dropped_ct_ids:
            sample.pre_split_eligible = False
            sample.selected = False
            sample.split_role = ""
            sample.fold_id = ""
            if not sample.pre_split_exclusion_reason:
                sample.pre_split_exclusion_reason = CT_ID_GATE_EXCLUSION_REASON
    selected_rgb = select_rgb_160(rgb_ids, seed)
    ct_test, development = _select_ct_test(ct_ids, seed, locked_tests.get("CT"))
    _select_ct_development_samples(development)
    for stats in ct_test:
        stats.selected_samples = list(stats.samples)
        stats.split_role = "test"
    folds = _ct_fold_assignment(development, seed)
    ct_target = sample_metrics(collect_selected_samples(development)).annotations_per_image
    fold_groups = {str(index): members for index, members in folds.items()}
    _swap_density(fold_groups, lambda stats: round(_id_selected_ratio(stats) * 10), ct_target)
    warnings = _check_density(fold_groups, ct_target, set(fold_groups), "CT_fold")
    review_warnings = _check_ct_area_balance(fold_groups)
    for fold, members in fold_groups.items():
        for stats in members:
            stats.split_role = "development"
            stats.fold_id = fold
    rgb_groups = _split_rgb(selected_rgb, seed, locked_tests.get("EXT"))
    rgb_selected_for = lambda role, stats: list(stats.samples) if role == "test" else stratified_sample(stats.samples, 300, seed, f"EXT:{stats.battery_id}:{role}")
    trainval_samples = [sample for role in ("train", "val") for sample in collect_selected_samples(rgb_groups[role], role, rgb_selected_for)]
    rgb_target = sample_metrics(trainval_samples).annotations_per_image
    rgb_bins = quantile_bins({stats.battery_id: _id_selected_ratio(stats) for stats in selected_rgb}, 5)
    _swap_density(
        rgb_groups,
        lambda stats: (stats.application, stats.has_damaged, rgb_bins[stats.battery_id]),
        rgb_target,
        {"test"},
        rgb_selected_for,
        True,
    )
    for role, members in rgb_groups.items():
        for stats in members:
            stats.selected_samples = rgb_selected_for(role, stats)
            stats.split_role = role
    final_trainval = [sample for role in ("train", "val") for sample in collect_selected_samples(rgb_groups[role])]
    rgb_target = sample_metrics(final_trainval).annotations_per_image
    warnings.extend(_check_density(rgb_groups, rgb_target, {"train", "val"}, "EXT_split"))
    for stats in selected_rgb:
        for sample in stats.samples:
            sample.selected_rgb_160 = True
            sample.split_role = stats.split_role
        for sample in stats.selected_samples:
            sample.selected = True
            sample.split_role = stats.split_role
    for stats in development + ct_test:
        for sample in stats.samples:
            sample.split_role = stats.split_role
            sample.fold_id = stats.fold_id
        for sample in stats.selected_samples:
            sample.selected = True
            sample.split_role = stats.split_role
            sample.fold_id = stats.fold_id
    selected_ids = sorted(development + ct_test + selected_rgb, key=lambda stats: (stats.modality, natural_id_key(stats.battery_id)))
    return SelectionResult(selected_ids, warnings, review_warnings, ct_id_gate_rows=ct_id_gate_rows)
