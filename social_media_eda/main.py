from __future__ import annotations

import json
from pathlib import Path

from data_loader import DataLoader
from model_trainer import ProductivityModelTrainer
from profiler import DataProfiler
from story_behavior_balance import StoryBehaviorBalance
from story_group_analysis import StoryGroupAnalysis
from story_relationships import StoryRelationships


def run_pipeline() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "social_media_productivity_6000.csv"

    output_dir = project_dir / "outputs"
    story_dir = output_dir / "stories"
    model_dir = output_dir / "models"
    profile_dir = output_dir / "profile"

    loader = DataLoader(data_path)
    loader.load()

    required = [
        "age",
        "daily_screen_time",
        "social_media_hours",
        "study_hours",
        "sleep_hours",
        "notifications_per_day",
        "focus_score",
        "addiction_level",
        "productivity_score",
    ]
    missing_columns = loader.validate_required_columns(required)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    pristine_df = loader.pandas_df

    profiler = DataProfiler(profile_dir)
    profile = profiler.build_profile(pristine_df)
    profiler.save_profile(profile)

    story_outputs = {
        "relationships": StoryRelationships(story_dir).run(pristine_df),
        "group_analysis": StoryGroupAnalysis(story_dir).run(pristine_df),
        "behavior_balance": StoryBehaviorBalance(story_dir).run(pristine_df),
    }

    trainer = ProductivityModelTrainer(model_dir, random_state=42)
    model_results = trainer.train_and_compare(pristine_df, target_col="productivity_score")

    summary = {
        "profile_rows": profile["rows"],
        "profile_columns": profile["columns"],
        "story_outputs": story_outputs,
        "model_results": model_results.to_dict(orient="records"),
    }

    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Pipeline complete.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    run_pipeline()
