from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ProductivityModelTrainer:
    """Train and compare baseline and tree-based models for productivity prediction."""

    def __init__(self, output_dir: str | Path, random_state: int = 42) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def _build_preprocessor(self, df: pd.DataFrame, target_col: str) -> tuple[ColumnTransformer, list[str], list[str]]:
        feature_cols = [c for c in df.columns if c != target_col]
        x = df[feature_cols]

        numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return preprocessor, feature_cols, categorical_features

    def train_and_compare(self, df: pd.DataFrame, target_col: str = "productivity_score") -> pd.DataFrame:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' is missing.")

        working = df.copy()
        working = working.dropna(subset=[target_col])

        preprocessor, feature_cols, _ = self._build_preprocessor(working, target_col)

        x = working[feature_cols]
        y = working[target_col]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=self.random_state,
        )

        models = {
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=250,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

        results = []
        for name, model in models.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipe.fit(x_train, y_train)
            preds = pipe.predict(x_test)

            rmse = root_mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            cv_scores = cross_val_score(
                pipe,
                x_train,
                y_train,
                cv=5,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse_mean = float((-cv_scores).mean())

            results.append(
                {
                    "model": name,
                    "test_rmse": round(float(rmse), 4),
                    "test_mae": round(float(mae), 4),
                    "test_r2": round(float(r2), 4),
                    "cv_rmse_mean": round(cv_rmse_mean, 4),
                }
            )

        result_df = pd.DataFrame(results).sort_values(by="test_rmse", ascending=True)
        path = self.output_dir / "model_comparison.csv"
        result_df.to_csv(path, index=False)

        return result_df
