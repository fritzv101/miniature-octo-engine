# miniature-octo-engine

From a compound to its targets, cell lines and cancer types.

`oncomap` is a Python backend that chains three questions together:

```
SMILES  ──▶  target binding (pActivity)  ──▶  cell line response (ln IC50)  ──▶  cancer-type summary
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,fetch]"
```

## Try it

```bash
oncomap demo --outdir artifacts/demo   # synthetic data -> train -> report
```

This writes a synthetic dataset, trains both stages, prints held-out metrics and
produces a compound report (targets, most sensitive cell lines, cancer types).

## Train on your own tables

```bash
oncomap train \
  --compounds data/compounds.csv \
  --binding data/binding.csv \
  --response data/response.csv \
  --cell-features data/expression.csv \
  --cell-lines data/cell_lines.csv \
  --model-out artifacts/pipeline.joblib

oncomap predict --model artifacts/pipeline.joblib --smiles "Cc1ccc2ncnc(Nc3ccccc3)c2c1"
```

Expected columns (override with `--schema chembl|gdsc`, or a `DatasetSchema` in code):

| table | required columns | notes |
|---|---|---|
| compounds | `compound_id`, `smiles` | optional `name` |
| binding | `compound_id`, `target_id`, `standard_value` | optional `standard_units`, `standard_type`, `standard_relation`, `smiles` |
| response | `compound_id`, `cell_line_id`, `ln_ic50` | lower = more sensitive; optional `auc` |
| cell_lines | `cell_line_id` | optional `cancer_type`, `tissue` |
| cell_features | `cell_line_id` + numeric columns | mutations / expression / CNV, any mix |

Units are normalized to pActivity = `-log10(M)`; `>`/`<` censored rows are dropped,
replicates are collapsed to a median, and SMILES are salt-stripped, neutralized
and canonicalized before anything is keyed on them.

## In code

```python
from oncomap import OncologyPipeline, load_training_data

data, reports = load_training_data(compounds, binding, response, cell_features, cell_lines)
pipeline = OncologyPipeline().fit(data)
report = pipeline.predict("Cc1ccc2ncnc(Nc3ccccc3)c2c1")
pipeline.save("artifacts/pipeline.joblib")
```

## How it works

**Stage 1 — binding** (`models/target_model.py`): one gradient-boosted regressor
per target over Morgan fingerprints, predicting pActivity. Targets are modelled
separately because bioactivity matrices are sparse. Every prediction carries the
maximum Tanimoto similarity to that target's training compounds, so predictions
outside the applicability domain are visible rather than silently trusted.

**Stage 2 — efficacy** (`models/response_model.py`): for each (compound, cell line)
pair the features are the stage-1 profile converted to fractional target
occupancy, physicochemical descriptors, an SVD compression of the cell line's
omics profile, and that cell line's mean response across training compounds.
A gradient-boosted tree learns which engagement-in-which-context is lethal.

**Honest evaluation** (`evaluation.py`): the binding stage is scored on a
Murcko-scaffold split (no chemical series shared between train and test) and the
efficacy stage on held-out *compounds*, which is the question that matters for a
new molecule. Random splits inflate both.

## Layout

```
src/oncomap/
  chem.py                 standardization, fingerprints, descriptors, scaffolds
  evaluation.py           scaffold/group splits, regression metrics
  pipeline.py             train, evaluate, predict, save/load
  cli.py                  oncomap demo|train|predict|evaluate
  data/schema.py          column mappings (default, ChEMBL, GDSC)
  data/loaders.py         cleaning, unit conversion, row accounting
  data/synthetic.py       synthetic compound->target->cell line dataset
  models/target_model.py  stage 1
  models/response_model.py stage 2
```

## Web API (for frontends)

```bash
pip install -e ".[api]"
ONCOMAP_MODEL=model.joblib ONCOMAP_COMPOUNDS=compounds.csv uvicorn oncomap.api:app --port 8000
# interactive docs: http://localhost:8000/docs
```

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/health` | | `{"status": "ok", "has_response_model": true}` |
| GET | `/model` | | target/cell-line counts and evaluation metrics |
| POST | `/predict` | `{"smiles": "...", "top_targets": 10, "top_cell_lines": 15}` | `canonical_smiles`, `descriptors`, `targets`, `cell_lines`, `cancer_types` |
| POST | `/predict/batch` | `{"smiles": ["...", "..."]}` | list of reports; invalid structures get `error` |
| GET | `/compounds?q=imatinib` | | known compounds (id, name, SMILES, MOA) |

Invalid SMILES return HTTP 422. CORS origins are set with `ONCOMAP_CORS_ORIGINS`
(comma-separated, default `*`). A `Dockerfile` is included: mount a trained model at
`/models/model.joblib`.

Set `ONCOMAP_API_KEY` to require a key on every endpoint except `/health`; clients
send it as `X-API-Key: <key>` or `Authorization: Bearer <key>`.

Deploy to a Hugging Face Docker Space (free CPU tier, 16 GB RAM):

```bash
pip install huggingface_hub
HF_TOKEN=hf_... python deploy/huggingface/deploy.py --space <user>/oncomap \
    --model model.joblib --compounds compounds.csv --api-key <key>
```

The key is stored as the Space secret `ONCOMAP_API_KEY`; the script prints the
public `https://<user>-oncomap.hf.space` URL.

## LINCS / PharmacoDB / DepMap data

For a response file with `pert_id,cell_line,sensitivity_score` (PharmacoDB DSS3
scores for LINCS compounds on DepMap ACH cell lines):

```bash
pip install -e ".[fetch]"
oncomap build-lincs-depmap --response L1000_..._dss3.csv --outdir data/lincs_depmap \
    [--expression OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv]
```

This fetches SMILES/MOA from LINCS `compoundinfo_beta.txt`, human binding activities
(pChEMBL) from the ChEMBL API, and lineage, hotspot mutations and expression from
DepMap on figshare, then prints the matching `oncomap train --schema lincs_depmap` command.

Binding tables may carry a `smiles` column: rows for compounds outside the compound
table (e.g. extra ChEMBL molecules for the same targets) are then kept and used to
train the binding model only. For bulk ChEMBL extraction restore the PostgreSQL dump
(`pg_restore --no-owner -t activities -t assays -t target_dictionary
-t molecule_dictionary -t compound_structures ...`) rather than paging the API.

Mechanistic response features: `--target-genes target_genes.csv` (columns
`target_chembl_id,gene_symbols`, genes `;`-separated) adds "occupancy of target x
expression / hotspot mutation of its gene" features. `--response-max-depth`,
`--response-max-iter` and `--response-min-samples-leaf` tune the response model; a
shallow, heavily regularized model (3 / 150 / 2000) generalizes better to unseen
compounds on the DSS3 data than the default.

## Caveats

* The synthetic dataset is for plumbing and tests, not for biology. Metrics on it
  say the pipeline learns the relationships it was given, nothing more.
* Only targets with enough measurements are modelled (`--min-samples-per-target`);
  skipped targets are recorded on the model.
* Response values are assumed to follow the ln IC50 convention (lower = more
  sensitive) unless `ResponseColumns.higher_is_more_sensitive` is set, as it is for
  DSS scores in the `lincs_depmap` schema. Rankings always put the most sensitive
  cell line first; `sensitivity_gain` > 0 means more sensitive than that cell
  line's typical compound.
