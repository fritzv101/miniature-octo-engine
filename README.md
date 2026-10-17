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

## Caveats

* The synthetic dataset is for plumbing and tests, not for biology. Metrics on it
  say the pipeline learns the relationships it was given, nothing more.
* Only targets with enough measurements are modelled (`--min-samples-per-target`);
  skipped targets are recorded on the model.
* Response values are assumed to follow the ln IC50 convention (lower = more
  sensitive). If yours are AUC or percent viability, keep the direction in mind
  when reading `delta_vs_baseline`.
