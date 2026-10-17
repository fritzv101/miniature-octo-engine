import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from oncomap import api  # noqa: E402
from oncomap.chem import FeaturizerConfig  # noqa: E402
from oncomap.pipeline import OncologyPipeline, PipelineConfig, load_training_data  # noqa: E402


@pytest.fixture(scope="module")
def client(dataset, tmp_path_factory):
    data, _ = load_training_data(
        dataset.compounds,
        dataset.binding,
        dataset.response,
        dataset.cell_features,
        dataset.cell_lines,
    )
    config = PipelineConfig()
    config.target.featurizer = FeaturizerConfig(fingerprint_bits=512, use_descriptors=False)
    config.target.min_samples_per_target = 10
    config.response.n_omics_components = 8
    tmp = tmp_path_factory.mktemp("api")
    model = OncologyPipeline(config).fit(data, evaluate=False).save(tmp / "model.joblib")
    compounds = tmp / "compounds.csv"
    dataset.compounds.to_csv(compounds, index=False)

    mp = pytest.MonkeyPatch()
    mp.setenv("ONCOMAP_MODEL", str(model))
    mp.setenv("ONCOMAP_COMPOUNDS", str(compounds))
    api.get_pipeline.cache_clear()
    api.get_compounds.cache_clear()
    yield TestClient(api.create_app())
    mp.undo()
    api.get_pipeline.cache_clear()
    api.get_compounds.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "has_response_model": True}


def test_predict_chains_targets_cells_and_cancers(client, dataset):
    smiles = dataset.compounds["smiles"].iloc[0]
    res = client.post("/predict", json={"smiles": smiles, "top_targets": 3, "top_cell_lines": 5})
    assert res.status_code == 200
    body = res.json()
    assert len(body["targets"]) == 3
    assert len(body["cell_lines"]) == 5
    assert body["cancer_types"]


def test_predict_rejects_invalid_smiles(client):
    assert client.post("/predict", json={"smiles": "not-a-molecule(("}).status_code == 422


def test_batch_isolates_bad_structures(client, dataset):
    res = client.post(
        "/predict/batch", json={"smiles": [dataset.compounds["smiles"].iloc[0], "(("]}
    )
    body = res.json()
    assert "targets" in body[0] and "error" in body[1]


def test_model_info_and_compound_search(client, dataset):
    assert client.get("/model").json()["n_targets"] > 0
    first = dataset.compounds["compound_id"].iloc[0]
    hits = client.get("/compounds", params={"q": first}).json()
    assert hits and hits[0]["compound_id"] == first
