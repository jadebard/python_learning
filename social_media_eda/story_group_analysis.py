from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class StoryGroupAnalysis:
    """Story 2: productivity and focus differences by addiction level groups."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> dict:
        if "addiction_level" not in df.columns:
            raise ValueError("Column 'addiction_level' is required for group analysis.")

        working = df.copy()
        working["addiction_level"] = working["addiction_level"].fillna("Unknown")

        boxplot_path = self.output_dir / "story2_productivity_by_addiction_level.png"
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=working, x="addiction_level", y="productivity_score")
        plt.title("Productivity Score by Addiction Level")
        plt.tight_layout()
        plt.savefig(boxplot_path, dpi=150)
        plt.close()

        mean_table = (
            working.groupby("addiction_level", dropna=False)[["productivity_score", "focus_score"]]
            .mean(numeric_only=True)
            .round(2)
            .sort_values("productivity_score", ascending=False)
        )
        table_path = self.output_dir / "story2_group_means.csv"
        mean_table.to_csv(table_path)

        return {
            "story": "group_analysis",
            "boxplot_path": str(boxplot_path),
            "group_mean_table_path": str(table_path),
            "group_means": mean_table.to_dict(),
        }
