import pandas as pd
from pathlib import Path

# BASE = Path("/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed")
BASE = Path("/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/final_aug_text")


sources = [
    # ("MELD_BONN_full.csv",                      "full_text"),
    ("MELD_BONN_hemisphere.csv",                "hemisphere_text"),
    # ("MELD_BONN_lobe_regions.csv",              "lobe_regions_text"),
    ("MELD_BONN_lobe.csv",                      "lobe_text"),
    # ("MELD_BONN_dominant_lobe.csv",             "dominant_lobe_text"),
    # ("MELD_BONN_hemisphere_lobe_regions.csv",   "hemisphere_lobe_regions_text"),
    ("MELD_BONN_hemisphere_lobe.csv",            "hemisphere_lobe_text"),
]

KEY_COLS = ["DATA_PATH", "ROI_PATH"]


def load_and_prepare(filename: str, new_text_col: str) -> pd.DataFrame:
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing_keys = [k for k in KEY_COLS if k not in df.columns]
    if missing_keys:
        raise ValueError(f"{filename}: missing required key columns {missing_keys}")

    # Identify potential text columns (excluding key columns)
    candidate_cols = [c for c in df.columns if c not in KEY_COLS and c != 'aal']
    print(candidate_cols)

    if not candidate_cols:
        # Create an empty text column if no source column is available
        df[new_text_col] = pd.NA
        return df[KEY_COLS + [new_text_col]]

    if len(candidate_cols) > 1:
        raise ValueError(
            f"{filename}: multiple non-key columns found {candidate_cols}. "
            f"Please specify which one should be used."
        )

    original_text_col = candidate_cols[0]
    df = df[KEY_COLS + [original_text_col]].copy()
    df = df.rename(columns={original_text_col: new_text_col})

    # Remove strict duplicates based on key columns (and text, if present)
    df = df.drop_duplicates(subset=KEY_COLS)
    return df


if __name__ == "__main__":
    # Load and merge all sources
    merged = None
    loaded_parts = {}

    for fname, new_col in sources:
        try:
            part = load_and_prepare(fname, new_col)
            loaded_parts[new_col] = part

            if merged is None:
                merged = part
            else:
                # Outer join to avoid losing subjects
                merged = merged.merge(part, on=KEY_COLS, how="outer")

            print(f"Added: {fname} -> {new_col}, shape={part.shape}")
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

    if merged is None:
        raise RuntimeError("Failed to assemble any input files.")

    merged["no_text"] = "full brain"

    # Optional: sort by subject_id if present
    if "subject_id" in merged.columns:
        merged = merged.sort_values("subject_id")

    # Final output path
    out_path = BASE / "MELD_BONN_mixed.csv"
    merged.to_csv(out_path, index=False)

    print(f"\nMerged file saved to: {out_path} (shape={merged.shape})")
    print("Columns:", list(merged.columns))
