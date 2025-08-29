import pandas as pd

from src.hf_loader import load_filter_to_dataframe
from src.wildchat_to_dataframe import rows_to_dataframe, basic_eda_summary


def test_rows_to_dataframe_structure():
    # Load a small non-stream subset then filter to 100 rows via pandas
    df = load_filter_to_dataframe(language='English', toxic=False, limit=100, split='train[:1000]')
    rows = df.to_dict(orient='records')
    df2 = rows_to_dataframe(rows)
    assert isinstance(df2, pd.DataFrame)
    assert len(df2) == 100


def test_basic_eda_summary_fields():
    rows = [{"a": 1, "b": None}, {"a": 2, "b": 3}]
    df = rows_to_dataframe(rows)
    summary = basic_eda_summary(df)
    assert summary["num_rows"] == 2
    assert summary["num_columns"] >= 2
    assert "columns" in summary and isinstance(summary["columns"], list)
    assert "dtypes" in summary and isinstance(summary["dtypes"], dict)
    assert "null_counts" in summary and isinstance(summary["null_counts"], dict) 