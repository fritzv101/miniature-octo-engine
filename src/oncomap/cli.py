"""Command line interface: ``oncomap demo|train|predict|evaluate``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from oncomap.data.loaders import load_cell_line_features, load_cell_lines, read_table
from oncomap.data.schema import CHEMBL_SCHEMA, GDSC_SCHEMA, LINCS_DEPMAP_SCHEMA, DatasetSchema
from oncomap.data.synthetic import write_synthetic_dataset
from oncomap.pipeline import OncologyPipeline, PipelineConfig, load_training_data

SCHEMAS: dict[str, DatasetSchema] = {
    "default": DatasetSchema(),
    "chembl": CHEMBL_SCHEMA,
    "gdsc": GDSC_SCHEMA,
    "lincs_depmap": LINCS_DEPMAP_SCHEMA,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oncomap", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="generate synthetic data, train and predict end to end")
    p_demo.add_argument("--outdir", default="artifacts/demo")
    p_demo.add_argument(
        "--smiles", default=None, help="compound to report on (default: a held-out analogue)"
    )

    p_train = sub.add_parser("train", help="train from local tables")
    p_train.add_argument("--compounds", required=True)
    p_train.add_argument("--binding", required=True)
    p_train.add_argument("--response")
    p_train.add_argument("--cell-features")
    p_train.add_argument("--cell-lines")
    p_train.add_argument("--schema", choices=sorted(SCHEMAS), default="default")
    p_train.add_argument("--model-out", default="artifacts/pipeline.joblib")
    p_train.add_argument("--metrics-out", default=None)
    p_train.add_argument("--min-samples-per-target", type=int, default=25)
    p_train.add_argument("--no-eval", action="store_true")

    p_pred = sub.add_parser("predict", help="report targets and cell lines for a compound")
    p_pred.add_argument("--model", default="artifacts/pipeline.joblib")
    p_pred.add_argument("--smiles", action="append", required=True)
    p_pred.add_argument("--top-targets", type=int, default=10)
    p_pred.add_argument("--top-cell-lines", type=int, default=15)
    p_pred.add_argument("--json-out", default=None)

    p_fetch = sub.add_parser(
        "build-lincs-depmap",
        help="fetch structures (LINCS), binding (ChEMBL) and cell-line data (DepMap)",
    )
    p_fetch.add_argument("--response", required=True, help="pert_id,cell_line,sensitivity_score")
    p_fetch.add_argument("--outdir", default="data/lincs_depmap")
    p_fetch.add_argument(
        "--expression", help="local DepMap/CCLE expression CSV (default: download 24Q4)"
    )

    p_eval = sub.add_parser("evaluate", help="print stored evaluation metrics of a model")
    p_eval.add_argument("--model", default="artifacts/pipeline.joblib")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    commands = {
        "demo": _demo,
        "train": _train,
        "predict": _predict,
        "evaluate": _evaluate,
        "build-lincs-depmap": _build_lincs_depmap,
    }
    return commands[args.command](args)


def _build_lincs_depmap(args: argparse.Namespace) -> int:
    from oncomap.data.sources import build_lincs_depmap_dataset

    paths = build_lincs_depmap_dataset(args.response, args.outdir, expression_path=args.expression)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(
        "next: oncomap train --schema lincs_depmap "
        f"--compounds {paths['compounds']} --binding {paths['binding']} "
        f"--response {args.response} --cell-lines {paths['cell_lines']} "
        f"--cell-features {paths['cell_features']}"
    )
    return 0


def _demo(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    paths = write_synthetic_dataset(str(outdir / "data"))
    print(f"wrote synthetic tables to {outdir / 'data'}")
    ns = argparse.Namespace(
        compounds=paths["compounds"],
        binding=paths["binding"],
        response=paths["response"],
        cell_features=paths["cell_features"],
        cell_lines=paths["cell_lines"],
        schema="default",
        model_out=str(outdir / "pipeline.joblib"),
        metrics_out=str(outdir / "metrics.json"),
        min_samples_per_target=25,
        no_eval=False,
    )
    _train(ns)
    smiles = args.smiles or pd.read_csv(paths["compounds"])["smiles"].iloc[0]
    return _predict(
        argparse.Namespace(
            model=ns.model_out,
            smiles=[smiles],
            top_targets=10,
            top_cell_lines=10,
            json_out=str(outdir / "report.json"),
        )
    )


def _train(args: argparse.Namespace) -> int:
    schema = SCHEMAS[args.schema]
    data, reports = load_training_data(
        read_table(args.compounds),
        read_table(args.binding),
        read_table(args.response) if args.response else None,
        _read_features(args.cell_features),
        load_cell_lines(read_table(args.cell_lines), schema.cell_lines)
        if args.cell_lines
        else None,
        schema=schema,
    )
    for line in reports:
        print(line)

    config = PipelineConfig()
    config.target.min_samples_per_target = args.min_samples_per_target
    config.response.higher_is_more_sensitive = schema.response.higher_is_more_sensitive
    pipeline = OncologyPipeline(config).fit(data, evaluate=not args.no_eval)
    path = pipeline.save(args.model_out)
    print(f"saved model to {path}")
    print(json.dumps(pipeline.metrics_summary(), indent=2, default=float))
    if args.metrics_out:
        pipeline.write_metrics(args.metrics_out)
    return 0


def _read_features(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    frame = read_table(path)
    id_col = frame.columns[0] if "cell_line_id" not in frame.columns else "cell_line_id"
    return load_cell_line_features(frame, id_col)


def _predict(args: argparse.Namespace) -> int:
    pipeline = OncologyPipeline.load(args.model)
    reports = pipeline.predict_batch(
        args.smiles, top_targets=args.top_targets, top_cell_lines=args.top_cell_lines
    )
    for report in reports:
        _print_report(report)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(reports, indent=2, default=float))
        print(f"\nwrote {args.json_out}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    pipeline = OncologyPipeline.load(args.model)
    print(json.dumps(pipeline.metrics_summary(), indent=2, default=float))
    return 0


def _print_report(report: dict[str, Any]) -> None:
    if "error" in report:
        print(f"\n{report['input_smiles']}: ERROR {report['error']}")
        return
    print(f"\n=== {report['canonical_smiles']} ===")
    d = report["descriptors"]
    print(
        f"MW {d['mol_weight']:.1f} | cLogP {d['clogp']:.2f} | TPSA {d['tpsa']:.1f} | "
        f"Lipinski violations {report['lipinski_violations']}"
    )
    print("\nPredicted targets:")
    print(
        pd.DataFrame(report["targets"])[
            ["target_name", "pactivity", "predicted_nm", "max_train_similarity", "in_domain"]
        ].to_string(index=False, float_format=lambda v: f"{v:,.2f}")
    )
    if report.get("cell_lines"):
        print("\nMost sensitive cell lines:")
        cols = [
            c
            for c in ("cell_line_id", "cancer_type", "predicted_response", "sensitivity_gain")
            if c in report["cell_lines"][0]
        ]
        print(
            pd.DataFrame(report["cell_lines"])[cols].to_string(
                index=False, float_format=lambda v: f"{v:,.2f}"
            )
        )
    if report.get("cancer_types"):
        print("\nBy cancer type:")
        print(
            pd.DataFrame(report["cancer_types"]).to_string(
                index=False, float_format=lambda v: f"{v:,.2f}"
            )
        )


if __name__ == "__main__":
    sys.exit(main())
