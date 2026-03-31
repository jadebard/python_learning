from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class StoryRelationships:
    """Story 1: relationships between behavioral features and productivity."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> dict:
        numeric_cols = [
            "age",
            "daily_screen_time",
            "social_media_hours",
            "study_hours",
            "sleep_hours",
            "notifications_per_day",
            "focus_score",
            "productivity_score",
        ]
        available = [c for c in numeric_cols if c in df.columns]
        corr = df[available].corr(numeric_only=True)

        heatmap_path = self.output_dir / "story1_correlation_heatmap.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        scatter_path = self.output_dir / "story1_study_vs_productivity.png"
        if {"study_hours", "productivity_score"}.issubset(df.columns):
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=df, x="study_hours", y="productivity_score", alpha=0.5)
            sns.regplot(data=df, x="study_hours", y="productivity_score", scatter=False, color="red")
            plt.title("Study Hours vs Productivity")
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=150)
            plt.close()

        top_productivity_corr = None
        if "productivity_score" in corr.columns:
            series = corr["productivity_score"].drop(labels=["productivity_score"], errors="ignore")
            if not series.empty:
                top_productivity_corr = series.sort_values(ascending=False).head(3).to_dict()

        return {
            "story": "relationships",
            "heatmap_path": str(heatmap_path),
            "scatter_path": str(scatter_path),
            "top_productivity_correlations": top_productivity_corr,
        }
