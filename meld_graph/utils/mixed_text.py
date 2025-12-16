import pandas as pd
from pathlib import Path
import sys

# BASE = Path("/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed")
BASE = Path("/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/final_aug_text")


sources = [
    # ("MELD_BONN_full.csv",                      "full_text"),
    ("MELD_BONN_hemisphere.csv",                "hemisphere_text"),
    # ("MELD_BONN_lobe_regions.csv",              "lobe_regions_text"),
    ("MELD_BONN_lobe.csv",              "lobe_text"),
    # ("MELD_BONN_dominant_lobe.csv", "dominant_lobe_text"),
    # ("MELD_BONN_hemisphere_lobe_regions.csv",   "hemisphere_lobe_regions_text"),
    ("MELD_BONN_hemisphere_lobe.csv",   "hemisphere_lobe_text"),
]

KEY_COLS = ["DATA_PATH", "ROI_PATH"]

def load_and_prepare(filename: str, new_text_col: str) -> pd.DataFrame:
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    
    missing_keys = [k for k in KEY_COLS if k not in df.columns]
    if missing_keys:
        raise ValueError(f"{filename}: отсутствуют ключевые колонки {missing_keys}")
    
    # Находим потенциальные текстовые колонки (кроме ключевых)
    candidate_cols = [c for c in df.columns if c not in KEY_COLS and c != 'aal']
    print(candidate_cols)
    if not candidate_cols:
        # Создадим пустую текстовую колонку, если нет источника
        df[new_text_col] = pd.NA
        return df[KEY_COLS + [new_text_col]]
    if len(candidate_cols) > 1:
        raise ValueError(
            f"{filename}: найдено несколько неключевых колонок {candidate_cols}. "
            f"Уточните, какую использовать."
        )
    
    original_text_col = candidate_cols[0]
    df = df[KEY_COLS + [original_text_col]].copy()
    df = df.rename(columns={original_text_col: new_text_col})

    # Удалим строгие дубли по ключу + тексту если вдруг
    df = df.drop_duplicates(subset=KEY_COLS)
    return df

if __name__ == "__main__":
    # Загружаем и объединяем
    merged = None
    loaded_parts = {}

    for fname, new_col in sources:
        try:
            part = load_and_prepare(fname, new_col)
            loaded_parts[new_col] = part
            if merged is None:
                merged = part
            else:
                # outer join, чтобы не потерять субъектов
                merged = merged.merge(part, on=KEY_COLS, how="outer")
            print(f"Добавлен: {fname} -> {new_col}, shape={part.shape}")
        except Exception as e:
            print(f"⚠️ Пропущен {fname}: {e}")

    if merged is None:
        raise RuntimeError("Не удалось собрать ни одного файла.")

    merged["no_text"] = "full brain"
    # Опционально: сортировка
    if "subject_id" in merged.columns:
        merged = merged.sort_values("subject_id")

    # Итоговый путь
    out_path = BASE / "MELD_BONN_mixed.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nСохранено объединённое: {out_path} (shape={merged.shape})")
    print("Колонки:", list(merged.columns))