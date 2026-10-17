import pandas as pd

from oncomap.data.sources import read_depmap_expression


def test_read_depmap_expression_new_layout_keeps_default_entries(tmp_path):
    path = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "SequencingID": ["CDS-1", "CDS-2", "CDS-3"],
            "ModelID": ["ACH-1", "ACH-1", "ACH-2"],
            "IsDefaultEntryForModel": ["No", "Yes", "Yes"],
            "TP53 (7157)": [1.0, 2.0, 3.0],
        }
    ).to_csv(path)
    expr = read_depmap_expression(path)
    assert list(expr.index) == ["ACH-1", "ACH-2"]
    assert expr.loc["ACH-1", "TP53 (7157)"] == 2.0


def test_read_depmap_expression_old_layout(tmp_path):
    path = tmp_path / "expr.csv"
    pd.DataFrame({"TP53 (7157)": [1.0]}, index=pd.Index(["ACH-1"])).to_csv(path)
    assert read_depmap_expression(path).loc["ACH-1", "TP53 (7157)"] == 1.0
