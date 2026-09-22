"""
Cleans NOAA's raw Marine Microplastics Database export into the schema
used by train_model.py.

Input:  data/raw/Marine_Microplastics.csv  (download from NOAA NCEI)
Output: data/processed/noaa_cleaned.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/raw/Marine_Microplastics.csv"
OUT_PATH = "data/processed/noaa_cleaned.csv"

INCOMPATIBLE_METHODS = ["Stainless steel spoon", "PVC cylinder", "Aluminum bucket"]


def clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    print(f"Raw records: {len(df)}")

    # Keep only the standard, comparable water-concentration unit
    df = df[df["Unit"] == "pieces/m3"].copy()
    print(f"After unit filter: {len(df)}")

    # Exclude sediment/bulk sampling tools incompatible with a volume unit
    # (values 1,000-10,000x higher than net-based methods -- not comparable)
    df = df[~df["Sampling Method"].isin(INCOMPATIBLE_METHODS)].copy()
    print(f"After sampling-method filter: {len(df)}")

    df = df.dropna(subset=["Latitude", "Longitude", "Measurement", "Oceans", "Date"])
    df["Date_parsed"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Date_parsed"])
    df["Year"] = df["Date_parsed"].dt.year
    df["Month"] = df["Date_parsed"].dt.month
    print(f"After date parsing: {len(df)}")

    df = df.rename(columns={
        "Oceans": "Ocean",
        "Sampling Method": "Sample_Method",
        "Measurement": "Concentration_pieces_m3",
    })

    threshold = float(np.quantile(df["Concentration_pieces_m3"], 0.75))
    df["Risk_Label"] = np.where(df["Concentration_pieces_m3"] > threshold, "High", "Low")

    final_cols = ["Latitude", "Longitude", "Year", "Month", "Ocean",
                  "Sample_Method", "Concentration_pieces_m3", "Risk_Label"]
    out = df[final_cols].reset_index(drop=True)

    print(f"\nFinal cleaned dataset: {len(out)} records")
    print(f"Elevated-risk threshold: {threshold:.4f} pieces/m3")
    print(out["Risk_Label"].value_counts())
    return out, threshold


if __name__ == "__main__":
    out, threshold = clean()
    out.to_csv(OUT_PATH, index=False)
    with open("data/processed/threshold.txt", "w") as f:
        f.write(str(threshold))
    print(f"\nSaved to {OUT_PATH}")
