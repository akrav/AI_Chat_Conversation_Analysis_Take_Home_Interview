from __future__ import annotations

from typing import List, Dict

import pandas as pd


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    return pd.json_normalize(rows)


def basic_eda_summary(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isna().sum().to_dict(),
    } 