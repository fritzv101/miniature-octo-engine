"""Stage 2: compound (+ predicted target profile) x cell line -> drug response.

The feature vector for a (compound, cell line) pair is deliberately mechanistic:

* the predicted target-engagement profile from stage 1, as fractional occupancy,
* a few interpretable physicochemical descriptors,
* a compressed view of the cell line's omics profile,
* the cell line's mean response across training compounds (how easy it is to
  kill at all) -- computed on training rows only, so it does not leak labels.

A gradient-boosted tree then learns the interactions (which omics context makes
engagement of which target lethal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from oncomap.chem import DESCRIPTOR_NAMES, descriptors
from oncomap.models.target_model import TargetBindingModel

logger = logging.getLogger(__name__)


@dataclass
class ResponseModelConfig:
    n_omics_components: int = 32
    occupancy_midpoint: float = 6.5  # pActivity giving 50% modelled occupancy
    occupancy_slope: float = 1.5
    use_descriptors: bool = True
    max_iter: int = 400
    learning_rate: float = 0.06
    max_depth: int = 7
    seed: int = 0


@dataclass
class CellLineResponseModel:
    """Predicts a response value (lower = more sensitive) per compound/cell line."""

    target_model: TargetBindingModel
    config: ResponseModelConfig = field(default_factory=ResponseModelConfig)

    def __post_init__(self) -> None:
        self.model: HistGradientBoostingRegressor | None = None
        self.cell_features: pd.DataFrame | None = None
        self.cell_lines: pd.DataFrame | None = None
        self._svd: TruncatedSVD | None = None
        self._scaler: StandardScaler | None = None
        self._omics: pd.DataFrame | None = None
        self._cell_baseline: pd.Series | None = None
        self._global_baseline: float = 0.0
        self.feature_names: list[str] = []

    def fit(
        self,
        response: pd.DataFrame,
        compounds: pd.DataFrame,
        cell_features: pd.DataFrame | None = None,
        cell_lines: pd.DataFrame | None = None,
    ) -> CellLineResponseModel:
        """``response``: compound_id, cell_line_id, response. ``compounds``: compound_id, smiles."""
        required = {"compound_id", "cell_line_id", "response"}
        missing = required - set(response.columns)
        if missing:
            raise ValueError(f"response frame missing columns: {sorted(missing)}")

        data = response.merge(compounds[["compound_id", "smiles"]], on="compound_id", how="inner")
        if data.empty:
            raise ValueError("no response rows matched the compound table on compound_id")
        self.cell_lines = cell_lines
        self._fit_omics(cell_features, data["cell_line_id"].unique())
        self._cell_baseline = data.groupby("cell_line_id")["response"].mean()
        self._global_baseline = float(data["response"].mean())

        x = self._build_features(data["smiles"].tolist(), data["cell_line_id"].tolist())
        y = data["response"].to_numpy(dtype=float)
        self.model = HistGradientBoostingRegressor(
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            l2_regularization=1.0,
            random_state=self.config.seed,
        ).fit(x, y)
        logger.info("fitted response model on %d pairs, %d features", len(y), x.shape[1])
        return self

    def predict(self, smiles: list[str], cell_line_ids: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        if len(smiles) != len(cell_line_ids):
            raise ValueError("smiles and cell_line_ids must have the same length")
        if not smiles:
            return np.zeros(0)
        return np.asarray(self.model.predict(self._build_features(smiles, cell_line_ids)))

    def predict_cell_lines(
        self, smiles: str, cell_line_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Ranked sensitivity of every known cell line to one compound."""
        ids = list(cell_line_ids or self.known_cell_lines)
        pred = self.predict([smiles] * len(ids), ids)
        out = pd.DataFrame({"cell_line_id": ids, "predicted_response": pred})
        baseline = (
            self._cell_baseline if self._cell_baseline is not None else pd.Series(dtype=float)
        )
        out["cell_line_baseline"] = baseline.reindex(ids).to_numpy()
        # Negative delta = more sensitive than that cell line's typical compound.
        out["delta_vs_baseline"] = out["predicted_response"] - out["cell_line_baseline"]
        if self.cell_lines is not None:
            meta = self.cell_lines.reindex(ids)
            for col in ("cancer_type", "tissue"):
                if col in meta.columns:
                    out[col] = meta[col].to_numpy()
        return out.sort_values("predicted_response", ignore_index=True)

    def summarize_by_cancer_type(self, smiles: str, min_lines: int = 3) -> pd.DataFrame:
        """Aggregate per-cell-line predictions into a cancer-type view."""
        pred = self.predict_cell_lines(smiles)
        if "cancer_type" not in pred.columns:
            raise ValueError("no cancer_type metadata was provided at fit time")
        grouped = pred.groupby("cancer_type").agg(
            n_cell_lines=("cell_line_id", "size"),
            median_response=("predicted_response", "median"),
            median_delta=("delta_vs_baseline", "median"),
            best_cell_line=("cell_line_id", "first"),
            best_response=("predicted_response", "min"),
        )
        grouped = grouped[grouped["n_cell_lines"] >= min_lines]
        return grouped.sort_values("median_delta").reset_index()

    @property
    def known_cell_lines(self) -> list[str]:
        if self._cell_baseline is None:
            return []
        return list(self._cell_baseline.index)

    def _fit_omics(self, cell_features: pd.DataFrame | None, cell_ids: np.ndarray) -> None:
        if cell_features is None or cell_features.empty:
            self._omics = None
            return
        features = cell_features.reindex(sorted(set(cell_ids) & set(cell_features.index)))
        if features.empty:
            logger.warning("no cell line in the response table has omics features; ignoring them")
            self._omics = None
            return
        self.cell_features = cell_features
        scaler = StandardScaler().fit(features.to_numpy(dtype=float))
        scaled = scaler.transform(features.to_numpy(dtype=float))
        n_components = min(self.config.n_omics_components, min(scaled.shape) - 1)
        if n_components < 1:
            self._omics = None
            return
        svd = TruncatedSVD(n_components=n_components, random_state=self.config.seed).fit(scaled)
        self._scaler, self._svd = scaler, svd
        self._omics = pd.DataFrame(
            svd.transform(scaler.transform(cell_features.to_numpy(dtype=float))),
            index=cell_features.index,
            columns=[f"omics_{i}" for i in range(n_components)],
        )

    def _build_features(self, smiles: list[str], cell_line_ids: list[str]) -> np.ndarray:
        unique = list(dict.fromkeys(smiles))
        profile = self.target_model.predict_matrix(unique)
        occupancy = 1 / (
            1
            + np.exp(
                -(profile.to_numpy() - self.config.occupancy_midpoint) * self.config.occupancy_slope
            )
        )
        occ_lookup = dict(zip(unique, occupancy, strict=True))
        blocks = [np.vstack([occ_lookup[s] for s in smiles])]
        names = [f"occ_{t}" for t in profile.columns]

        if self.config.use_descriptors:
            desc_lookup = {s: np.array(list(descriptors(s).values()), dtype=float) for s in unique}
            blocks.append(np.vstack([desc_lookup[s] for s in smiles]))
            names.extend(DESCRIPTOR_NAMES)

        n_rows = len(smiles)
        if self._omics is not None:
            omics = self._omics.reindex(cell_line_ids)
            blocks.append(np.nan_to_num(omics.to_numpy(dtype=float)))
            names.extend(self._omics.columns)

        baseline = (
            self._cell_baseline.reindex(cell_line_ids).fillna(self._global_baseline).to_numpy()
            if self._cell_baseline is not None
            else np.full(n_rows, self._global_baseline)
        )
        blocks.append(baseline.reshape(-1, 1))
        names.append("cell_line_baseline")
        self.feature_names = list(names)
        return np.hstack(blocks).astype(np.float32)
