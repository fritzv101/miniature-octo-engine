import numpy as np
import pandas as pd
import pytest

from oncomap.data.loaders import (
    load_binding,
    load_cell_line_features,
    load_compounds,
    load_response,
    nanomolar_to_pactivity,
)
from oncomap.data.schema import BindingColumns, ResponseColumns, SchemaError


def test_load_compounds_cleans_and_dedupes():
    frame = pd.DataFrame(
        {
            "compound_id": ["A", "B", "C", "A"],
            "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "garbage((", None, "CCO"],
        }
    )
    out, report = load_compounds(frame)
    assert list(out["compound_id"]) == ["A"]
    assert report.dropped == {"missing_smiles": 1, "invalid_smiles": 1, "duplicate_id": 1}


def test_missing_column_raises_schema_error():
    with pytest.raises(SchemaError, match="smiles"):
        load_compounds(pd.DataFrame({"compound_id": ["A"]}))


def test_pactivity_conversion():
    assert nanomolar_to_pactivity(1.0) == pytest.approx(9.0)
    assert nanomolar_to_pactivity(1000.0) == pytest.approx(6.0)


def test_load_binding_units_censoring_and_replicates():
    frame = pd.DataFrame(
        {
            "compound_id": ["A", "A", "A", "A"],
            "smiles": ["CCO"] * 4,
            "target_id": ["T1", "T1", "T2", "T3"],
            "standard_type": ["IC50", "IC50", "IC50", "pIC50"],
            "standard_relation": ["=", "=", ">", "="],
            "standard_value": [100.0, 1.0, 50.0, 7.5],
            "standard_units": ["nM", "uM", "nM", None],
        }
    )
    out, report = load_binding(frame)
    by_target = out.set_index("target_id")["pactivity"]
    # 100 nM -> 7.0 and 1 uM -> 6.0, median 6.5
    assert by_target["T1"] == pytest.approx(6.5)
    assert by_target["T3"] == pytest.approx(7.5)
    assert "T2" not in by_target.index
    assert report.dropped["censored"] == 1


def test_load_binding_restricts_to_known_compounds():
    compounds = pd.DataFrame({"compound_id": ["A"], "smiles": ["CCO"], "name": ["a"]})
    frame = pd.DataFrame(
        {
            "compound_id": ["A", "ZZ"],
            "target_id": ["T1", "T1"],
            "standard_type": ["pIC50", "pIC50"],
            "standard_value": [7.0, 8.0],
        }
    )
    out, report = load_binding(frame, BindingColumns(smiles=None), compounds=compounds)
    assert list(out["compound_id"]) == ["A"]
    assert report.dropped["unknown_compound"] == 1


def test_load_response_medians_replicates():
    frame = pd.DataFrame(
        {
            "compound_id": ["A", "A", "B"],
            "cell_line_id": ["C1", "C1", "C1"],
            "ln_ic50": [1.0, 3.0, np.nan],
        }
    )
    out, report = load_response(frame)
    assert out.loc[0, "response"] == pytest.approx(2.0)
    assert report.dropped["missing_value"] == 1


def test_cell_line_features_drop_sparse_and_constant_columns():
    frame = pd.DataFrame(
        {
            "cell_line_id": ["C1", "C2", "C3", "C4"],
            "good": [1.0, 2.0, 3.0, 4.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "sparse": [1.0, None, None, None],
        }
    )
    out = load_cell_line_features(frame, prefix="expr_")
    assert list(out.columns) == ["expr_good"]
    assert out.index.name == "cell_line_id"


def test_load_response_log1p_transform():
    frame = pd.DataFrame({"pert_id": ["A", "B"], "cell_line": ["C1", "C1"], "dss": [0.0, np.e - 1]})
    cols = ResponseColumns(
        compound_id="pert_id", cell_line_id="cell_line", value="dss", auc=None, transform="log1p"
    )
    out, _ = load_response(frame, cols)
    assert out.set_index("compound_id")["response"].to_dict() == pytest.approx({"A": 0.0, "B": 1.0})
