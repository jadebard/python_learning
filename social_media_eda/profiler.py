from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class DataProfiler:
    """Generate a lightweight profile report for data quality and scope."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_profile(self, df: pd.DataFrame) -> dict:
        missing_counts = df.isna().sum().to_dict()
        dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        numeric_ranges = {}
        for col in numeric_cols:
            numeric_ranges[col] = {
                "min": float(df[col].min(skipna=True)) if df[col].notna().any() else None,
                "max": float(df[col].max(skipna=True)) if df[col].notna().any() else None,
                "mean": float(df[col].mean(skipna=True)) if df[col].notna().any() else None,
            }

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "dtypes": dtypes,
            "missing_counts": missing_counts,
            "missing_percent": {
                col: round((count / len(df)) * 100, 2) if len(df) else 0.0
                for col, count in missing_counts.items()
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_ranges": numeric_ranges,
        }

    def save_profile(self, profile: dict, filename: str = "profile_report.json") -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        return path
