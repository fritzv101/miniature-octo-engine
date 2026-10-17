import numpy as np
import pytest

from oncomap.chem import FeaturizerConfig
from oncomap.data.loaders import load_binding
from oncomap.evaluation import regression_metrics, scaffold_split
from oncomap.models.response_model import CellLineResponseModel, ResponseModelConfig
from oncomap.models.target_model import TargetBindingModel, TargetModelConfig


@pytest.fixture(scope="module")
def binding(dataset):
    out, _ = load_binding(dataset.binding)
    return out


@pytest.fixture(scope="module")
def fitted_target_model(binding):
    config = TargetModelConfig(
        featurizer=FeaturizerConfig(fingerprint_bits=512, use_descriptors=False),
        min_samples_per_target=10,
    )
    return TargetBindingModel(config).fit(binding)


def test_target_model_learns_structure_activity(binding, fitted_target_model):
    train, test = scaffold_split(binding["smiles"].tolist(), 0.25, seed=5)
    model = TargetBindingModel(fitted_target_model.config).fit(binding.iloc[train])
    preds = model.evaluate(binding.iloc[test])
    metrics = regression_metrics(preds["pactivity"].to_numpy(), preds["predicted"].to_numpy())
    assert metrics["r2"] > 0.2, metrics
    assert metrics["rmse"] < 1.0, metrics


def test_target_model_requires_minimum_samples(binding):
    config = TargetModelConfig(min_samples_per_target=10**9)
    with pytest.raises(ValueError, match="min_samples_per_target"):
        TargetBindingModel(config).fit(binding)


def test_predict_profile_is_ranked_and_annotated(fitted_target_model, dataset):
    smiles = dataset.compounds["smiles"].iloc[0]
    profile = fitted_target_model.predict_profile(smiles)
    assert list(profile["pactivity"]) == sorted(profile["pactivity"], reverse=True)
    assert profile["max_train_similarity"].between(0, 1).all()
    assert profile["predicted_nm"].gt(0).all()
    assert len(fitted_target_model.predict_profile(smiles, top_k=3)) == 3


def test_similarity_flags_out_of_domain_compounds(fitted_target_model):
    familiar = fitted_target_model.predict_profile("CNc1ncnc2cccc(C)c12")
    alien = fitted_target_model.predict_profile("[Na+].[Cl-].CCCCCCCCCCCCCCCC")
    assert familiar["max_train_similarity"].max() > alien["max_train_similarity"].max()


def test_response_model_predicts_unseen_compounds(dataset, fitted_target_model):
    response = dataset.response.rename(columns={"ln_ic50": "response"})
    held_out = set(dataset.compounds["compound_id"].iloc[:40])
    train = response[~response["compound_id"].isin(held_out)]
    test = response[response["compound_id"].isin(held_out)].merge(
        dataset.compounds[["compound_id", "smiles"]], on="compound_id"
    )
    model = CellLineResponseModel(
        fitted_target_model, ResponseModelConfig(n_omics_components=8)
    ).fit(train, dataset.compounds, dataset.cell_features, dataset.cell_lines)
    pred = model.predict(test["smiles"].tolist(), test["cell_line_id"].tolist())
    metrics = regression_metrics(test["response"].to_numpy(), pred)
    assert metrics["r2"] > 0.2, metrics


def test_response_model_rankings_and_cancer_type_summary(dataset, fitted_target_model):
    response = dataset.response.rename(columns={"ln_ic50": "response"})
    model = CellLineResponseModel(
        fitted_target_model, ResponseModelConfig(n_omics_components=8)
    ).fit(response, dataset.compounds, dataset.cell_features, dataset.cell_lines)
    smiles = dataset.compounds["smiles"].iloc[1]
    ranked = model.predict_cell_lines(smiles)
    assert len(ranked) == len(model.known_cell_lines)
    assert ranked["predicted_response"].is_monotonic_increasing
    assert "cancer_type" in ranked.columns

    summary = model.summarize_by_cancer_type(smiles, min_lines=1)
    assert summary["median_sensitivity_gain"].is_monotonic_decreasing
    assert summary["n_cell_lines"].sum() == len(ranked)


def test_response_model_works_without_omics(dataset, fitted_target_model):
    response = dataset.response.rename(columns={"ln_ic50": "response"})
    model = CellLineResponseModel(fitted_target_model).fit(response, dataset.compounds)
    pred = model.predict([dataset.compounds["smiles"].iloc[0]] * 3, model.known_cell_lines[:3])
    assert np.isfinite(pred).all()


def test_response_model_rejects_mismatched_inputs(dataset, fitted_target_model):
    response = dataset.response.rename(columns={"ln_ic50": "response"})
    model = CellLineResponseModel(fitted_target_model).fit(response, dataset.compounds)
    with pytest.raises(ValueError, match="same length"):
        model.predict(["CCO"], ["a", "b"])


def test_higher_is_more_sensitive_flips_rankings(dataset, fitted_target_model):
    response = dataset.response.rename(columns={"ln_ic50": "response"})
    flipped = response.assign(response=-response["response"])
    model = CellLineResponseModel(
        fitted_target_model, ResponseModelConfig(higher_is_more_sensitive=True)
    ).fit(flipped, dataset.compounds)
    ranked = model.predict_cell_lines(dataset.compounds["smiles"].iloc[1])
    assert ranked["predicted_response"].is_monotonic_decreasing
