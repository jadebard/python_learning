from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DataLoader:
    """Load and expose pristine dataset copies in pandas and NumPy formats."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._df: Optional[pd.DataFrame] = None
        self._np: Optional[np.ndarray] = None

    def load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self._df = df.copy(deep=True)
        self._np = df.to_numpy(copy=True)

    @property
    def pandas_df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Dataset has not been loaded. Call load() first.")
        return self._df.copy(deep=True)

    @property
    def numpy_array(self) -> np.ndarray:
        if self._np is None:
            raise RuntimeError("Dataset has not been loaded. Call load() first.")
        return self._np.copy()

    def validate_required_columns(self, required_columns: list[str]) -> list[str]:
        df = self.pandas_df
        missing = [col for col in required_columns if col not in df.columns]
        return missing
