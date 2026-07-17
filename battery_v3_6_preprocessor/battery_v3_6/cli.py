from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import approve_selection, dry_run, execute
from .training_view import build_training_view


def _common_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed (default: 42)")


def _jobs_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", type=int, default=1, help="worker processes for scan/output (default: 1 = serial). Determinism is preserved.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="battery-v3.6", description="Battery preprocessing v3.6")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run", help="scan, validate, select, and write reports only")
    dry.add_argument("--raw-root", type=Path, required=True)
    dry.add_argument("--work-dir", type=Path, required=True)
    _common_seed(dry)
    _jobs_arg(dry)
    approve = subparsers.add_parser("approve-selection", help="promote candidate selection after review")
    approve.add_argument("--work-dir", type=Path, required=True)
    approve.add_argument("--approved-by", required=True)
    _common_seed(approve)
    run = subparsers.add_parser("execute", help="generate the approved dataset through staging")
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--keep-failed-staging", action="store_true")
    _common_seed(run)
    _jobs_arg(run)
    view = subparsers.add_parser("training-view", help="build an Ultralytics images/labels/data.yaml view")
    _add_training_view_args(view)
    return parser


def _add_training_view_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modality", choices=("CT", "EXT"), required=True)
    parser.add_argument("--task", choices=("det", "seg"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--copy", action="store_true", help="copy instead of hardlink")


def training_view_main(default_dataset_root: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a YOLO training view")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modality", choices=("CT", "EXT"), required=True)
    parser.add_argument("--task", choices=("det", "seg"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--copy", action="store_true", help="copy instead of hardlink")
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error("--dataset-root is required")
    if args.output is None:
        fold_suffix = f"_fold{args.fold}" if args.modality == "CT" else ""
        args.output = args.dataset_root / "training_views" / f"{args.modality}_{args.task}{fold_suffix}"
    build_training_view(args.dataset_root, args.output, args.modality, args.task, args.fold, args.include_test, args.copy)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "dry-run":
        _, warnings = dry_run(args.raw_root, args.work_dir, args.seed, args.jobs)
        print(f"dry-run complete; quality warnings={len(warnings)}")
    elif args.command == "approve-selection":
        approve_selection(args.work_dir, args.approved_by, args.seed)
        print("selection approved")
    elif args.command == "execute":
        execute(args.raw_root, args.work_dir, args.output, args.seed, args.keep_failed_staging, args.jobs)
        print(f"dataset created: {args.output}")
    elif args.command == "training-view":
        build_training_view(args.dataset_root, args.output, args.modality, args.task, args.fold, args.include_test, args.copy)


if __name__ == "__main__":
    main()

