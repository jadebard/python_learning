from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class StoryBehaviorBalance:
    """Story 3: compare high-screen and low-screen users on study/sleep/productivity."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> dict:
        if "daily_screen_time" not in df.columns:
            raise ValueError("Column 'daily_screen_time' is required for behavior balance story.")

        working = df.copy()
        median_screen = working["daily_screen_time"].median(skipna=True)
        working["screen_group"] = working["daily_screen_time"].apply(
            lambda x: "HighScreen" if pd.notna(x) and x >= median_screen else "LowScreen"
        )

        compare_cols = ["study_hours", "sleep_hours", "productivity_score"]
        summary = working.groupby("screen_group")[compare_cols].mean(numeric_only=True).round(2)
        summary_path = self.output_dir / "story3_screen_group_summary.csv"
        summary.to_csv(summary_path)

        plot_path = self.output_dir / "story3_screen_group_productivity.png"
        plt.figure(figsize=(7, 5))
        sns.barplot(data=working, x="screen_group", y="productivity_score")
        plt.title("Average Productivity: High vs Low Screen Users")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return {
            "story": "behavior_balance",
            "median_screen_time": float(median_screen),
            "summary_path": str(summary_path),
            "plot_path": str(plot_path),
            "summary": summary.to_dict(),
        }
