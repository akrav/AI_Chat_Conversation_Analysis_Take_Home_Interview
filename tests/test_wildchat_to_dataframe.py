import pandas as pd

from src.wildchat_loader import load_wildchat_subset
from src.wildchat_to_dataframe import rows_to_dataframe, basic_eda_summary


def test_rows_to_dataframe_structure():
    rows = load_wildchat_subset(sample_size=5, prefer_http=True)
    df = rows_to_dataframe(rows)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_basic_eda_summary_fields():
    rows = [{"a": 1, "b": None}, {"a": 2, "b": 3}]
    df = rows_to_dataframe(rows)
    summary = basic_eda_summary(df)
    assert summary["num_rows"] == 2
    assert summary["num_columns"] >= 2
    assert "columns" in summary and isinstance(summary["columns"], list)
    assert "dtypes" in summary and isinstance(summary["dtypes"], dict)
    assert "null_counts" in summary and isinstance(summary["null_counts"], dict) 