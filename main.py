# main.py
from __future__ import annotations

import argparse

import classification
import regression
import feature_selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised ML Interview Workbench")
    p.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "classification", "regression", "feature_selection"],
        help="What to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.task in ("all", "classification"):
        print("- RUNNING CLASSIFICATION -")
        classification.main()

    if args.task in ("all", "regression"):
        print("- RUNNING REGRESSION -")
        regression.main()

    if args.task in ("all", "feature_selection"):
        print("- RUNNING FEATURE SELECTION -")
        feature_selection.run_feature_selection_experiments()


if __name__ == "__main__":
    main()

"""
Starten im Terminal mit:
python main.py --task classification
python main.py --task regression
python main.py --task feature_selection
python main.py --task all
"""
