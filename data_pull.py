from __future__ import annotations

import os
import pandas as pd

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "statcan_impact_panel.parquet")

FILES = [
    ("Q2", os.path.join(DATA_DIR, "q2-tariff-impact", "33100992.csv")),
    ("Q3", os.path.join(DATA_DIR, "q3-tariff-impact", "33101028.csv")),
    ("Q4", os.path.join(DATA_DIR, "q4-tariff-impact", "33101069.csv")),
]


def normalize_perspective(question: str) -> str:
    q = str(question).lower()
    if "u.s. tariffs" in q or "us tariffs" in q:
        return "U.S. tariffs on goods sold (exports)"
    if "canadian tariffs" in q or "canada tariffs" in q:
        return "Canadian tariffs on goods purchased (imports)"
    return "Other / unspecified"


def impact_weight(level: str) -> int:
    l = str(level).strip().lower()

    if "no impact" in l:
        return 0
    if "low" in l or "minor" in l:
        return 1
    if "moderate" in l or "medium" in l:
        return 2
    if "high" in l or "major" in l or "severe" in l:
        return 3

    # exclude these from the index
    if "not applicable" in l or "don't know" in l or "dont know" in l or "unknown" in l:
        return -1

    return -1


def find_question_col(df: pd.DataFrame) -> str:
    # In your file it contains this substring
    for c in df.columns:
        if "Level of impact of tariffs or trade barriers" in c:
            return c
    # fallback: try fuzzy match
    candidates = [c for c in df.columns if "level of impact" in c.lower() and "tariff" in c.lower()]
    if not candidates:
        raise RuntimeError("Could not find the impact question column.")
    return sorted(candidates, key=len, reverse=True)[0]


def load_one(quarter: str, path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    qcol = find_question_col(df)

    required = ["REF_DATE", "GEO", "Business characteristics", "Level impact", "VALUE", "UOM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")

    out = df[[c for c in [
        "REF_DATE", "GEO", "DGUID", "Business characteristics",
        qcol, "Level impact", "UOM", "VALUE", "STATUS", "SYMBOL"
    ] if c in df.columns]].copy()

    out.rename(columns={qcol: "Question", "Level impact": "Impact_level"}, inplace=True)

    out["Quarter"] = quarter
    out["REF_DATE"] = out["REF_DATE"].astype(str).str.strip()
    out["Period"] = out["REF_DATE"] + " " + out["Quarter"]  # "2025 Q3"
    out["GEO"] = out["GEO"].astype(str).str.strip()
    out["Business characteristics"] = out["Business characteristics"].astype(str).str.strip()
    out["Impact_level"] = out["Impact_level"].astype(str).str.strip()
    out["UOM"] = out["UOM"].astype(str).str.strip()

    out["Perspective"] = out["Question"].map(normalize_perspective)
    out["Impact_weight"] = out["Impact_level"].map(impact_weight)

    out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce")
    out = out.dropna(subset=["VALUE"]).copy()

    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    frames = []
    for quarter, path in FILES:
        print(f"Loading {quarter}: {path}")
        frames.append(load_one(quarter, path))

    panel = pd.concat(frames, ignore_index=True)

    # Order Period nicely
    period_order = ["2025 Q2", "2025 Q3", "2025 Q4"]
    # If your REF_DATE changes in future, this still works:
    unique_periods = panel["Period"].unique().tolist()
    if set(period_order).issubset(set(unique_periods)):
        panel["Period"] = pd.Categorical(panel["Period"], categories=period_order, ordered=True)

    panel.to_parquet(OUT_FILE, index=False)

    print("\nSaved:", OUT_FILE)
    print("Rows:", len(panel))
    print("Periods:", sorted(panel["Period"].astype(str).unique().tolist()))
    print("Perspectives:", panel["Perspective"].unique())
    print("Impact levels sample:", panel["Impact_level"].unique()[:8])


if __name__ == "__main__":
    main()
