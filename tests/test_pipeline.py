import json

import pytest

from oncomap.chem import FeaturizerConfig
from oncomap.cli import main
from oncomap.pipeline import OncologyPipeline, PipelineConfig, load_training_data


@pytest.fixture(scope="module")
def fitted_pipeline(dataset):
    data, reports = load_training_data(
        dataset.compounds,
        dataset.binding,
        dataset.response.rename(columns={"ln_ic50": "ln_ic50"}),
        dataset.cell_features,
        dataset.cell_lines,
    )
    assert any("binding" in r for r in reports)
    config = PipelineConfig()
    config.target.featurizer = FeaturizerConfig(fingerprint_bits=512, use_descriptors=False)
    config.target.min_samples_per_target = 10
    config.response.n_omics_components = 8
    return OncologyPipeline(config).fit(data)


def test_pipeline_reports_both_stage_metrics(fitted_pipeline):
    summary = fitted_pipeline.metrics_summary()
    assert summary["target"]["r2"] > 0.2
    assert summary["response"]["r2"] > 0.2
    assert summary["response"]["split"] == "unseen_compound"


def test_predict_returns_chained_report(fitted_pipeline, dataset):
    report = fitted_pipeline.predict(dataset.compounds["smiles"].iloc[2], top_targets=3)
    assert report["canonical_smiles"]
    assert len(report["targets"]) == 3
    assert report["cell_lines"] and report["cancer_types"]
    assert (
        report["cell_lines"][0]["predicted_response"]
        <= report["cell_lines"][-1]["predicted_response"]
    )


def test_predict_batch_isolates_bad_structures(fitted_pipeline, dataset):
    reports = fitted_pipeline.predict_batch([dataset.compounds["smiles"].iloc[0], "nope(("])
    assert "targets" in reports[0]
    assert "error" in reports[1]


def test_save_and_load_roundtrip(fitted_pipeline, dataset, tmp_path):
    path = fitted_pipeline.save(tmp_path / "pipeline.joblib")
    reloaded = OncologyPipeline.load(path)
    smiles = dataset.compounds["smiles"].iloc[3]
    before = fitted_pipeline.predict(smiles)["targets"][0]["pactivity"]
    after = reloaded.predict(smiles)["targets"][0]["pactivity"]
    assert before == pytest.approx(after)
    assert reloaded.metrics_summary() == fitted_pipeline.metrics_summary()


def test_cli_demo_end_to_end(tmp_path, capsys):
    outdir = tmp_path / "demo"
    assert main(["demo", "--outdir", str(outdir)]) == 0
    captured = capsys.readouterr().out
    assert "Predicted targets" in captured
    assert "By cancer type" in captured
    report = json.loads((outdir / "report.json").read_text())
    assert report[0]["targets"]
    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["target"]["n"] > 0
