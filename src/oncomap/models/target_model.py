"""Stage 1: compound structure -> protein target binding affinity.

One regressor per target over the compound's fingerprint. Targets are modelled
separately because bioactivity matrices are very sparse: a given compound has
been measured against a handful of targets, so a shared multi-output model would
spend most of its capacity on missing cells. A nearest-neighbour Tanimoto
similarity to the target's training set is reported alongside each prediction as
an applicability-domain flag -- a confident-looking number for a compound unlike
anything in training is the main failure mode of this kind of model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from oncomap.chem import FeaturizerConfig, MoleculeFeaturizer

logger = logging.getLogger(__name__)

ACTIVE_PACTIVITY = 6.0  # 1 uM, the usual "this compound hits the target" line


def _make_estimator(kind: str, seed: int) -> Any:
    if kind == "gbdt":
        return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=seed
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if kind == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"unknown estimator kind: {kind!r}")


@dataclass
class TargetModelConfig:
    estimator: str = "gbdt"
    min_samples_per_target: int = 25
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    seed: int = 0


@dataclass
class TargetBindingModel:
    """Predicts pActivity for every target it was trained on."""

    config: TargetModelConfig = field(default_factory=TargetModelConfig)

    def __post_init__(self) -> None:
        self.featurizer = MoleculeFeaturizer(self.config.featurizer)
        self.models: dict[str, Any] = {}
        self.target_names: dict[str, str] = {}
        self.training_counts: dict[str, int] = {}
        self.target_means: dict[str, float] = {}
        self._train_fps: dict[str, np.ndarray] = {}
        self.skipped_targets: dict[str, int] = {}

    @property
    def targets(self) -> list[str]:
        return list(self.models)

    def fit(self, binding: pd.DataFrame) -> TargetBindingModel:
        """``binding`` needs columns: compound_id, smiles, target_id, pactivity."""
        required = {"smiles", "target_id", "pactivity"}
        missing = required - set(binding.columns)
        if missing:
            raise ValueError(f"binding frame missing columns: {sorted(missing)}")

        unique_smiles = binding["smiles"].drop_duplicates().tolist()
        feature_cache = dict(
            zip(unique_smiles, self.featurizer.featurize_many(unique_smiles), strict=True)
        )
        for target_id, sub in binding.groupby("target_id"):
            if len(sub) < self.config.min_samples_per_target:
                self.skipped_targets[str(target_id)] = len(sub)
                continue
            x = np.vstack([feature_cache[s] for s in sub["smiles"]])
            y = sub["pactivity"].to_numpy(dtype=float)
            model = _make_estimator(self.config.estimator, self.config.seed)
            model.fit(x, y)
            key = str(target_id)
            self.models[key] = model
            self.training_counts[key] = len(sub)
            self.target_means[key] = float(y.mean())
            self._train_fps[key] = _binary_fp(x, self.config.featurizer.fingerprint_bits)
            if "target_name" in sub.columns:
                self.target_names[key] = str(sub["target_name"].iloc[0])
        logger.info(
            "trained %d targets (%d skipped for <%d samples)",
            len(self.models),
            len(self.skipped_targets),
            self.config.min_samples_per_target,
        )
        if not self.models:
            raise ValueError(
                "no target had enough measurements; lower min_samples_per_target "
                f"(currently {self.config.min_samples_per_target})"
            )
        return self

    def predict_matrix(self, smiles: Sequence[str]) -> pd.DataFrame:
        """Compounds x targets pActivity matrix (rows follow the input order)."""
        smiles = list(smiles)
        if not smiles:
            return pd.DataFrame(columns=self.targets, dtype=float)
        x = self.featurizer.featurize_many(smiles)
        data = {t: np.asarray(m.predict(x), dtype=np.float64) for t, m in self.models.items()}
        return pd.DataFrame(data, index=pd.Index(smiles, name="smiles"))

    def similarity_matrix(self, smiles: Sequence[str]) -> pd.DataFrame:
        """Max Tanimoto similarity to each target's training compounds."""
        smiles = list(smiles)
        fps = _binary_fp(
            self.featurizer.featurize_many(smiles), self.config.featurizer.fingerprint_bits
        )
        data = {t: _max_tanimoto(fps, train) for t, train in self._train_fps.items()}
        return pd.DataFrame(data, index=pd.Index(smiles, name="smiles"))

    def predict_profile(self, smiles: str, *, top_k: int | None = None) -> pd.DataFrame:
        """Ranked target profile for one compound, with confidence annotations."""
        pred = self.predict_matrix([smiles]).iloc[0]
        sim = self.similarity_matrix([smiles]).iloc[0]
        out = pd.DataFrame(
            {
                "target_id": pred.index,
                "target_name": [self.target_names.get(t, t) for t in pred.index],
                "pactivity": pred.to_numpy(),
                "predicted_nm": 10 ** (9 - pred.to_numpy()),
                "is_active": pred.to_numpy() >= ACTIVE_PACTIVITY,
                "max_train_similarity": sim.reindex(pred.index).to_numpy(),
                "training_compounds": [self.training_counts[t] for t in pred.index],
            }
        ).sort_values("pactivity", ascending=False, ignore_index=True)
        out["in_domain"] = out["max_train_similarity"] >= 0.3
        return out.head(top_k) if top_k else out

    def evaluate(self, binding: pd.DataFrame) -> pd.DataFrame:
        """Per-row predictions on a held-out binding frame (for metrics)."""
        known = binding[binding["target_id"].astype(str).isin(self.models)].copy()
        if known.empty:
            return known.assign(predicted=[], max_train_similarity=[])
        matrix = self.predict_matrix(known["smiles"].drop_duplicates().tolist())
        sims = self.similarity_matrix(matrix.index.tolist())
        known["predicted"] = [
            matrix.at[s, str(t)] for s, t in zip(known["smiles"], known["target_id"], strict=True)
        ]
        known["max_train_similarity"] = [
            sims.at[s, str(t)] for s, t in zip(known["smiles"], known["target_id"], strict=True)
        ]
        return known


def _binary_fp(features: np.ndarray, n_bits: int) -> np.ndarray:
    return (features[:, :n_bits] > 0).astype(np.float32)


def _max_tanimoto(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.zeros(len(query))
    inter = query @ reference.T
    q = query.sum(1)[:, None]
    r = reference.sum(1)[None, :]
    union = q + r - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(union > 0, inter / union, 0.0)
    return tan.max(axis=1)
