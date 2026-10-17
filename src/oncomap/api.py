"""HTTP API for frontends: ``uvicorn oncomap.api:app``.

Environment:
    ONCOMAP_MODEL         path to a model saved by ``oncomap train`` (required)
    ONCOMAP_COMPOUNDS     optional compounds CSV (id, name, smiles) for /compounds search
    ONCOMAP_CORS_ORIGINS  comma-separated allowed origins (default ``*``)
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from oncomap.chem import InvalidSmilesError
from oncomap.pipeline import OncologyPipeline


class PredictRequest(BaseModel):
    smiles: str = Field(..., examples=["CC(=O)Oc1ccccc1C(=O)O"])
    top_targets: int = Field(10, ge=1, le=200)
    top_cell_lines: int = Field(15, ge=1, le=2000)


class BatchPredictRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, max_length=100)
    top_targets: int = Field(10, ge=1, le=200)
    top_cell_lines: int = Field(15, ge=1, le=2000)


def _clean(obj: Any) -> Any:
    """Make numpy/pandas values JSON-safe (NaN/inf -> null)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


@lru_cache(maxsize=1)
def get_pipeline() -> OncologyPipeline:
    path = os.environ.get("ONCOMAP_MODEL")
    if not path:
        raise RuntimeError("set ONCOMAP_MODEL to a trained model file")
    return OncologyPipeline.load(path)


@lru_cache(maxsize=1)
def get_compounds() -> pd.DataFrame:
    path = os.environ.get("ONCOMAP_COMPOUNDS")
    if not path:
        return pd.DataFrame(columns=["compound_id", "name", "smiles"])
    df = pd.read_csv(path)
    rename = {"pert_id": "compound_id", "cmap_name": "name"}
    return df.rename(columns=rename)


def create_app() -> FastAPI:
    app = FastAPI(
        title="oncomap",
        version="0.1.0",
        description="Compound (SMILES) -> protein targets -> cancer cell lines -> cancer types",
    )
    origins = [o.strip() for o in os.environ.get("ONCOMAP_CORS_ORIGINS", "*").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            pipe = get_pipeline()
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
        return {"status": "ok", "has_response_model": pipe.response_model is not None}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        pipe = get_pipeline()
        assert pipe.target_model is not None
        metrics = {
            stage: {k: v for k, v in m.items() if not k.startswith("per_")}
            for stage, m in pipe.metrics.items()
            if isinstance(m, dict)
        }
        info: dict[str, Any] = {
            "n_targets": len(pipe.target_model.targets),
            "metrics": metrics,
            "higher_is_more_sensitive": pipe.config.response.higher_is_more_sensitive,
        }
        if pipe.response_model is not None:
            info["n_cell_lines"] = len(pipe.response_model.known_cell_lines)
        return _clean(info)

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        try:
            report = get_pipeline().predict(
                req.smiles, top_targets=req.top_targets, top_cell_lines=req.top_cell_lines
            )
        except (InvalidSmilesError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid SMILES: {exc}") from exc
        return _clean(report)

    @app.post("/predict/batch")
    def predict_batch(req: BatchPredictRequest) -> list[dict[str, Any]]:
        reports = get_pipeline().predict_batch(
            req.smiles, top_targets=req.top_targets, top_cell_lines=req.top_cell_lines
        )
        return _clean(reports)

    @app.get("/compounds")
    def compounds(
        q: str = Query("", description="name or id substring"),
        limit: int = Query(20, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        df = get_compounds()
        if q:
            hay = df["compound_id"].astype(str) + " " + df.get("name", "").astype(str)
            df = df[hay.str.contains(q, case=False, regex=False)]
        cols = [c for c in ("compound_id", "name", "smiles", "moa", "annotated_targets") if c in df]
        return _clean(df[cols].head(limit).to_dict("records"))

    return app


app = create_app()
