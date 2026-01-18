import math
import os
import warnings

import pandas as pd

REQUIRED_COLUMNS = ("filepath", "label")


def read_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Manifest {csv_path} missing required column(s): {missing_str}")

    labels = pd.to_numeric(df["label"], errors="raise")
    if labels.isna().any():
        raise ValueError(f"Manifest {csv_path} has missing values in 'label'.")
    if not labels.isin([0, 1]).all():
        bad_values = sorted(set(labels[~labels.isin([0, 1])].tolist()))
        raise ValueError(
            f"Manifest {csv_path} column 'label' must be int 0/1. Found: {bad_values}"
        )

    df = df.copy()
    df["label"] = labels.astype(int)

    paths = df["filepath"].astype(str).tolist()
    missing_count = sum(1 for path in paths if not os.path.exists(path))
    if missing_count:
        warnings.warn(
            f"{missing_count} filepath(s) do not exist locally. These manifests were likely "
            "generated on Kaggle; paths will not resolve on Windows. You can regenerate "
            "manifests locally later or run training in Kaggle.",
            UserWarning,
        )

    return df


def manifest_stats(df: pd.DataFrame) -> dict:
    n_rows = int(len(df))
    n_pos = int(df["label"].sum()) if n_rows else 0
    n_neg = n_rows - n_pos
    pos_frac = (n_pos / n_rows) if n_rows else 0.0
    return {"n_rows": n_rows, "n_pos": n_pos, "n_neg": n_neg, "pos_frac": pos_frac}


def print_manifest_report(name: str, df: pd.DataFrame) -> None:
    stats = manifest_stats(df)
    print(f"{name} manifest")
    print(f"  rows: {stats['n_rows']}")
    print(f"  pos: {stats['n_pos']}")
    print(f"  neg: {stats['n_neg']}")
    print(f"  pos_frac: {stats['pos_frac']:.4f}")


def sample_preview(df: pd.DataFrame, n: int = 6, title: str = "Sample") -> None:
    paths = df["filepath"].astype(str).head(n).tolist()
    existing = [path for path in paths if os.path.exists(path)]

    if not existing:
        print(
            "Sample preview unavailable: these manifests were created on Kaggle and the "
            "filepaths are Kaggle paths. Consider re-generating manifests locally later or "
            "adapting training to run in Kaggle."
        )
        return

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    cols = min(3, len(existing))
    rows = int(math.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if hasattr(axes, "ravel"):
        flat_axes = list(axes.ravel())
    else:
        flat_axes = [axes]

    for ax, path in zip(flat_axes, existing):
        try:
            image = mpimg.imread(path)
            ax.imshow(image)
            ax.set_title(os.path.basename(path))
        except Exception:
            ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(path)}", ha="center")
            ax.set_title("Error")
        ax.axis("off")

    for ax in flat_axes[len(existing) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
