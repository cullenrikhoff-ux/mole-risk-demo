import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Ensure the repo root is on sys.path when run as a script.
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.config import TEST_MANIFEST, TRAIN_MANIFEST, VAL_MANIFEST
from training.manifest_utils import manifest_stats, print_manifest_report, read_manifest


def write_summary(path: str, stats_by_split: dict) -> None:
    lines = [
        "# Manifest Report",
        "",
        "| split | n_rows | n_pos | n_neg | pos_frac |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split in ("train", "val", "test"):
        stats = stats_by_split[split]
        lines.append(
            f"| {split} | {stats['n_rows']} | {stats['n_pos']} | {stats['n_neg']} | "
            f"{stats['pos_frac']:.4f} |"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    manifests = {
        "train": TRAIN_MANIFEST,
        "val": VAL_MANIFEST,
        "test": TEST_MANIFEST,
    }
    stats_by_split = {}

    for name, path in manifests.items():
        df = read_manifest(path)
        print_manifest_report(name, df)
        stats_by_split[name] = manifest_stats(df)

    write_summary("results/manifest_report.md", stats_by_split)
    print(
        "Day 2 complete: manifests ready. Next: Day 3 training notebook reads these CSVs in Kaggle."
    )


if __name__ == "__main__":
    main()
