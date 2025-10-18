"""Basic tests for evaluation utilities."""

from pathlib import Path

import pandas as pd

from src.credit_risk_agent import evaluate_risk


def test_evaluate_risk_adds_error_column(tmp_path: Path) -> None:
    """Ensure evaluate_risk adds reconstruction error column."""
    data = pd.DataFrame(
        {
            "age": [30, 40],
            "income": [50000, 75000],
            "loan_amount": [10000, 20000],
            "defaulted": [0, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)
    result = evaluate_risk(csv_path)
    assert "reconstruction_error" in result.columns
