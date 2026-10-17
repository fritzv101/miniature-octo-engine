"""End-to-end pipeline: SMILES -> targets -> cell lines -> cancer types.

``OncologyPipeline`` wires the two model stages together, owns train/evaluate/
persist, and exposes a single :meth:`predict` entry point that answers the
question the project is about: given a structure, what does it hit, and which
tumour models should respond.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from oncomap.chem import descriptors, lipinski_violations, standardize_smiles
from oncomap.data.loaders import load_binding, load_compounds, load_response
from oncomap.data.schema import DatasetSchema
from oncomap.evaluation import group_split, per_group_metrics, regression_metrics, scaffold_split
from oncomap.models.response_model import CellLineResponseModel, ResponseModelConfig
from oncomap.models.target_model import TargetBindingModel, TargetModelConfig

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1


@dataclass
class PipelineConfig:
    target: TargetModelConfig = field(default_factory=TargetModelConfig)
    response: ResponseModelConfig = field(default_factory=ResponseModelConfig)
    test_fraction: float = 0.2
    seed: int = 0


@dataclass
class TrainingData:
    """Cleaned inputs for training, as produced by the loaders."""

    compounds: pd.DataFrame
    binding: pd.DataFrame
    response: pd.DataFrame | None = None
    cell_features: pd.DataFrame | None = None
    cell_lines: pd.DataFrame | None = None


class OncologyPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.target_model: TargetBindingModel | None = None
        self.response_model: CellLineResponseModel | None = None
        self.metrics: dict[str, Any] = {}

    # ----------------------------------------------------------------- training
    def fit(self, data: TrainingData, *, evaluate: bool = True) -> OncologyPipeline:
        binding = data.binding
        if evaluate:
            train_idx, test_idx = scaffold_split(
                binding["smiles"].tolist(), self.config.test_fraction, self.config.seed
            )
            held_out_binding = binding.iloc[test_idx]
            fit_binding = binding.iloc[train_idx]
        else:
            fit_binding, held_out_binding = binding, binding.iloc[0:0]

        self.target_model = TargetBindingModel(self.config.target).fit(fit_binding)
        if evaluate and len(held_out_binding):
            self.metrics["target"] = self._evaluate_targets(held_out_binding)

        if data.response is not None and len(data.response):
            self.response_model = CellLineResponseModel(self.target_model, self.config.response)
            if evaluate:
                self.metrics["response"] = self._evaluate_response(data)
            self.response_model.fit(
                data.response, data.compounds, data.cell_features, data.cell_lines
            )
        return self

    def _evaluate_targets(self, held_out: pd.DataFrame) -> dict[str, Any]:
        assert self.target_model is not None
        preds = self.target_model.evaluate(held_out)
        metrics: dict[str, Any] = dict(
            regression_metrics(preds["pactivity"].to_numpy(), preds["predicted"].to_numpy())
        )
        in_domain = preds[preds["max_train_similarity"] >= 0.3]
        metrics["in_domain"] = regression_metrics(
            in_domain["pactivity"].to_numpy(), in_domain["predicted"].to_numpy()
        )
        metrics["per_target"] = per_group_metrics(
            preds, "target_id", "pactivity", "predicted"
        ).to_dict("records")
        logger.info(
            "target model (scaffold split): %s", {k: metrics[k] for k in ("n", "rmse", "r2")}
        )
        return metrics

    def _evaluate_response(self, data: TrainingData) -> dict[str, Any]:
        """Hold out whole compounds: the realistic question is a brand-new molecule."""
        assert self.target_model is not None
        response = data.response
        assert response is not None
        train_idx, test_idx = group_split(
            response["compound_id"].tolist(), self.config.test_fraction, self.config.seed
        )
        if len(test_idx) == 0:
            return {}
        train = response.iloc[train_idx]
        test_ids = set(response.iloc[test_idx]["compound_id"])
        # Held-out compounds must be unseen by the binding model too, or their
        # measured affinities leak into the response features.
        binding = data.binding[~data.binding["compound_id"].isin(test_ids)]
        target_model = TargetBindingModel(self.config.target).fit(binding)
        probe = CellLineResponseModel(target_model, self.config.response).fit(
            train, data.compounds, data.cell_features, data.cell_lines
        )
        test = response.iloc[test_idx].merge(
            data.compounds[["compound_id", "smiles"]], on="compound_id", how="inner"
        )
        test = test.assign(
            predicted=probe.predict(test["smiles"].tolist(), test["cell_line_id"].tolist())
        )
        y = test["response"].to_numpy()
        metrics: dict[str, Any] = dict(regression_metrics(y, test["predicted"].to_numpy()))
        metrics["split"] = "unseen_compound"
        cell_mean = train.groupby("cell_line_id")["response"].mean()
        baseline = test["cell_line_id"].map(cell_mean).fillna(train["response"].mean())
        metrics["baseline_cell_line_mean"] = regression_metrics(y, baseline.to_numpy())
        metrics["compound_level"] = _compound_level_metrics(test)
        metrics["per_cell_line"] = per_group_metrics(
            test, "cell_line_id", "response", "predicted"
        ).to_dict("records")
        logger.info(
            "response model (unseen compounds): %s",
            {k: metrics.get(k) for k in ("n", "rmse", "r2")},
        )
        return metrics

    # ---------------------------------------------------------------- inference
    def predict(
        self, smiles: str, *, top_targets: int = 10, top_cell_lines: int = 15
    ) -> dict[str, Any]:
        """Full compound report: chemistry, target profile, sensitive models, tumour types."""
        if self.target_model is None:
            raise RuntimeError("pipeline is not fitted")
        canonical = standardize_smiles(smiles)
        result: dict[str, Any] = {
            "input_smiles": smiles,
            "canonical_smiles": canonical,
            "descriptors": descriptors(canonical),
            "lipinski_violations": lipinski_violations(canonical),
            "targets": self.target_model.predict_profile(canonical, top_k=top_targets).to_dict(
                "records"
            ),
        }
        if self.response_model is not None:
            cells = self.response_model.predict_cell_lines(canonical)
            result["cell_lines"] = cells.head(top_cell_lines).to_dict("records")
            try:
                result["cancer_types"] = self.response_model.summarize_by_cancer_type(
                    canonical
                ).to_dict("records")
            except ValueError:
                result["cancer_types"] = []
        return result

    def predict_batch(self, smiles: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        out = []
        for s in smiles:
            try:
                out.append(self.predict(s, **kwargs))
            except ValueError as exc:  # invalid structure should not kill a batch
                out.append({"input_smiles": s, "error": str(exc)})
        return out

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "config": asdict(self.config),
                "target_model": self.target_model,
                "response_model": self.response_model,
                "metrics": self.metrics,
            },
            path,
            compress=3,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> OncologyPipeline:
        payload = joblib.load(Path(path))
        version = payload.get("version")
        if version != ARTIFACT_VERSION:
            raise ValueError(f"artifact version {version} != expected {ARTIFACT_VERSION}")
        pipeline = cls()
        pipeline.target_model = payload["target_model"]
        pipeline.response_model = payload["response_model"]
        pipeline.metrics = payload.get("metrics", {})
        return pipeline

    def metrics_summary(self) -> dict[str, Any]:
        """Headline numbers only, safe to print or dump as JSON."""
        keep = ("n", "rmse", "mae", "r2", "pearson", "spearman", "split")
        nested = ("in_domain", "baseline_cell_line_mean", "compound_level")
        summary: dict[str, Any] = {}
        for stage, values in self.metrics.items():
            summary[stage] = {k: v for k, v in values.items() if k in keep}
            for key in nested:
                if isinstance(values.get(key), dict):
                    summary[stage][key] = {
                        k: v for k, v in values[key].items() if k in keep or k not in nested
                    }
        return summary

    def write_metrics(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.metrics_summary(), indent=2, default=float))
        return path


def load_training_data(
    compounds: pd.DataFrame,
    binding: pd.DataFrame,
    response: pd.DataFrame | None = None,
    cell_features: pd.DataFrame | None = None,
    cell_lines: pd.DataFrame | None = None,
    schema: DatasetSchema | None = None,
) -> tuple[TrainingData, list[str]]:
    """Run raw frames through the cleaning loaders; returns data plus load reports."""
    schema = schema or DatasetSchema()
    compounds_clean, c_report = load_compounds(compounds, schema.compounds)
    binding_clean, b_report = load_binding(binding, schema.binding, compounds=compounds_clean)
    reports = [str(c_report), str(b_report)]
    response_clean = None
    if response is not None:
        response_clean, r_report = load_response(response, schema.response)
        known = set(compounds_clean["compound_id"])
        before = len(response_clean)
        response_clean = response_clean[response_clean["compound_id"].isin(known)]
        r_report.drop("unknown_compound", before - len(response_clean))
        r_report.rows_out = len(response_clean)
        reports.append(str(r_report))
    return (
        TrainingData(compounds_clean, binding_clean, response_clean, cell_features, cell_lines),
        reports,
    )


def _compound_level_metrics(test: pd.DataFrame) -> dict[str, float]:
    """Does the model rank compounds by potency, and cell lines within a compound?"""
    per_compound = test.groupby("compound_id")[["response", "predicted"]].mean()
    within = [
        g["response"].corr(g["predicted"], method="spearman")
        for _, g in test.groupby("compound_id")
        if len(g) >= 10 and g["response"].std() > 0 and g["predicted"].std() > 0
    ]
    return {
        "n_compounds": float(len(per_compound)),
        "mean_response_spearman": float(
            per_compound["response"].corr(per_compound["predicted"], method="spearman")
        ),
        "within_compound_spearman_median": float(np.median(within)) if within else float("nan"),
    }
