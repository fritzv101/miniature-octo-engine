"""Load and clean compound, binding, response and cell-line tables.

Each loader returns a tidy frame with fixed column names so models never need
to know where the data came from:

* compounds:  compound_id, smiles, name
* binding:    compound_id, smiles, target_id, pactivity
* response:   compound_id, cell_line_id, response
* cell lines: cell_line_id (index), cancer_type, tissue
* features:   cell_line_id (index) x numeric omics columns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from oncomap.chem import try_standardize_smiles
from oncomap.data.schema import (
    BindingColumns,
    CellLineColumns,
    CompoundColumns,
    ResponseColumns,
    optional_column,
    require_columns,
)

logger = logging.getLogger(__name__)

# Multipliers that convert a concentration to nanomolar.
_TO_NM: dict[str, float] = {
    "pm": 1e-3,
    "nm": 1.0,
    "um": 1e3,
    "µm": 1e3,
    "μm": 1e3,
    "mm": 1e6,
    "m": 1e9,
}

POTENCY_TYPES = frozenset({"IC50", "KI", "KD", "EC50", "GI50", "XC50", "AC50"})
NEG_LOG_TYPES = frozenset({"PIC50", "PKI", "PKD", "PEC50", "PCHEMBL", "PCHEMBL_VALUE"})


@dataclass
class LoadReport:
    """Row accounting for a load, so silent data loss is visible."""

    source: str
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.dropped.items()) or "none"
        return f"{self.source}: {self.rows_in} in -> {self.rows_out} out (dropped: {parts})"


def read_table(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read CSV/TSV/Parquet/Excel based on file extension."""
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, **kwargs)  # type: ignore[arg-type]
    if suffixes.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs)  # type: ignore[arg-type]
    if suffixes.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t", **kwargs)  # type: ignore[arg-type]
    return pd.read_csv(path, **kwargs)  # type: ignore[arg-type]


def nanomolar_to_pactivity(value_nm: np.ndarray | pd.Series | float) -> np.ndarray:
    """Convert a potency in nM to -log10(M), the pChEMBL convention."""
    arr = np.asarray(value_nm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 9.0 - np.log10(arr)


def to_nanomolar(values: pd.Series, units: pd.Series) -> pd.Series:
    factors = units.astype(str).str.strip().str.lower().map(_TO_NM)
    return pd.to_numeric(values, errors="coerce") * factors


def load_compounds(
    frame: pd.DataFrame, columns: CompoundColumns | None = None
) -> tuple[pd.DataFrame, LoadReport]:
    cols = columns or CompoundColumns()
    require_columns(frame, [cols.compound_id, cols.smiles], what="compound")
    report = LoadReport("compounds", rows_in=len(frame))

    name_col = optional_column(frame, cols.name)
    out = pd.DataFrame(
        {
            "compound_id": frame[cols.compound_id].astype(str),
            "raw_smiles": frame[cols.smiles],
            "name": frame[name_col].astype(str)
            if name_col
            else frame[cols.compound_id].astype(str),
        }
    )
    missing = out["raw_smiles"].isna()
    report.drop("missing_smiles", missing.sum())
    out = out[~missing]

    out["smiles"] = out["raw_smiles"].astype(str).map(try_standardize_smiles)
    bad = out["smiles"].isna()
    report.drop("invalid_smiles", bad.sum())
    out = out[~bad]

    dup = out.duplicated("compound_id")
    report.drop("duplicate_id", dup.sum())
    out = out[~dup].drop(columns="raw_smiles").reset_index(drop=True)
    report.rows_out = len(out)
    logger.info("%s", report)
    return out, report


def load_binding(
    frame: pd.DataFrame,
    columns: BindingColumns | None = None,
    *,
    compounds: pd.DataFrame | None = None,
    keep_censored: bool = False,
    min_pactivity: float = 2.0,
    max_pactivity: float = 13.0,
) -> tuple[pd.DataFrame, LoadReport]:
    """Normalize a bioactivity table to one median pActivity per (compound, target).

    Accepts raw concentrations (IC50/Ki/Kd... with units) or values that are
    already -log10 (pIC50, pChEMBL). Censored values ('>' / '<') are dropped by
    default because treating them as exact biases the regression.
    """
    cols = columns or BindingColumns()
    require_columns(frame, [cols.compound_id, cols.target_id, cols.value], what="binding")
    report = LoadReport("binding", rows_in=len(frame))
    df = frame.copy()

    relation_col = optional_column(df, cols.relation)
    if relation_col and not keep_censored:
        rel = df[relation_col].astype(str).str.strip().str.strip("'\"")
        censored = ~rel.isin(["=", "nan", "None", ""])
        report.drop("censored", censored.sum())
        df = df[~censored]

    assay_col = optional_column(df, cols.assay_type)
    units_col = optional_column(df, cols.units)
    assay = (
        df[assay_col].astype(str).str.upper().str.replace(" ", "", regex=False)
        if assay_col
        else pd.Series("IC50", index=df.index)
    )
    values = pd.to_numeric(df[cols.value], errors="coerce")

    is_neglog = assay.isin(NEG_LOG_TYPES)
    is_potency = assay.isin(POTENCY_TYPES)
    report.drop("unsupported_assay_type", (~(is_neglog | is_potency)).sum())

    pact = pd.Series(np.nan, index=df.index, dtype=float)
    pact[is_neglog] = values[is_neglog]
    if is_potency.any():
        if units_col:
            nm = to_nanomolar(values[is_potency], df.loc[is_potency, units_col])
        else:
            nm = values[is_potency]
        pact[is_potency] = nanomolar_to_pactivity(nm)

    df = df.assign(pactivity=pact)
    invalid = ~np.isfinite(df["pactivity"]) & (is_neglog | is_potency)
    report.drop("unconvertible_value", invalid.sum())
    df = df[np.isfinite(df["pactivity"])]

    out_of_range = ~df["pactivity"].between(min_pactivity, max_pactivity)
    report.drop("out_of_range", out_of_range.sum())
    df = df[~out_of_range]

    df = df.rename(columns={cols.compound_id: "compound_id", cols.target_id: "target_id"})
    df["compound_id"] = df["compound_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)

    smiles_col = optional_column(frame, cols.smiles)
    if compounds is not None:
        # Known compounds take the compound table's structure; other rows (e.g. extra
        # ChEMBL molecules used only to train the binding model) need their own SMILES.
        own = df.pop(smiles_col) if smiles_col else None
        df = df.drop(columns=[c for c in ("smiles",) if c in df.columns])
        before = len(df)
        df = df.merge(compounds[["compound_id", "smiles"]], on="compound_id", how="left")
        if own is not None:
            extra = df["smiles"].isna().to_numpy()
            raw = own.to_numpy()[extra].astype(str)
            canonical = {s: try_standardize_smiles(s) for s in set(raw)}
            df.loc[extra, "smiles"] = [canonical[s] for s in raw]
        df = df[df["smiles"].notna()]
        report.drop("unknown_compound", before - len(df))
    elif smiles_col:
        df["smiles"] = df[smiles_col].astype(str).map(try_standardize_smiles)
        bad = df["smiles"].isna()
        report.drop("invalid_smiles", bad.sum())
        df = df[~bad]
    else:
        raise ValueError("binding table has no SMILES column and no compound table was given")

    keys = ["compound_id", "smiles", "target_id"]
    name_col = optional_column(frame, cols.target_name)
    agg = df.groupby(keys, as_index=False)["pactivity"].median()
    if name_col:
        names = df.groupby("target_id")[name_col].first().rename("target_name")
        agg = agg.merge(names, left_on="target_id", right_index=True, how="left")
    report.drop("merged_replicates", len(df) - len(agg))
    report.rows_out = len(agg)
    logger.info("%s", report)
    return agg.reset_index(drop=True), report


def load_response(
    frame: pd.DataFrame, columns: ResponseColumns | None = None
) -> tuple[pd.DataFrame, LoadReport]:
    """Normalize a drug-response table to one median value per (compound, cell line).

    The direction of ``response`` is whatever the source uses; see
    ``ResponseColumns.higher_is_more_sensitive``.
    """
    cols = columns or ResponseColumns()
    require_columns(frame, [cols.compound_id, cols.cell_line_id, cols.value], what="response")
    report = LoadReport("response", rows_in=len(frame))
    df = pd.DataFrame(
        {
            "compound_id": frame[cols.compound_id].astype(str),
            "cell_line_id": frame[cols.cell_line_id].astype(str),
            "response": pd.to_numeric(frame[cols.value], errors="coerce"),
        }
    )
    auc_col = optional_column(frame, cols.auc)
    if auc_col:
        df["auc"] = pd.to_numeric(frame[auc_col], errors="coerce")
    bad = ~np.isfinite(df["response"])
    report.drop("missing_value", bad.sum())
    df = df[~bad]
    if cols.transform == "log1p":
        negative = df["response"] < 0
        report.drop("negative_for_log1p", negative.sum())
        df = df[~negative].assign(response=lambda d: np.log1p(d["response"]))
    elif cols.transform is not None:
        raise ValueError(f"unknown response transform: {cols.transform!r}")
    agg_cols = {"response": "median", **({"auc": "median"} if auc_col else {})}
    agg = df.groupby(["compound_id", "cell_line_id"], as_index=False).agg(agg_cols)
    report.drop("merged_replicates", len(df) - len(agg))
    report.rows_out = len(agg)
    logger.info("%s", report)
    return agg, report


def load_cell_lines(frame: pd.DataFrame, columns: CellLineColumns | None = None) -> pd.DataFrame:
    cols = columns or CellLineColumns()
    require_columns(frame, [cols.cell_line_id], what="cell line")
    out = pd.DataFrame({"cell_line_id": frame[cols.cell_line_id].astype(str)})
    for target, source in (("cancer_type", cols.cancer_type), ("tissue", cols.tissue)):
        src = optional_column(frame, source)
        out[target] = frame[src].astype(str).values if src else "unknown"
    return out.drop_duplicates("cell_line_id").set_index("cell_line_id")


def load_cell_line_features(
    frame: pd.DataFrame,
    id_column: str = "cell_line_id",
    *,
    prefix: str | None = None,
    max_missing_fraction: float = 0.5,
) -> pd.DataFrame:
    """Numeric omics matrix (cell lines x features).

    Columns with too many missing values are dropped; the rest are median-imputed.
    ``prefix`` namespaces the columns (e.g. ``mut_``, ``expr_``) when several omics
    blocks are concatenated.
    """
    require_columns(frame, [id_column], what="cell line features")
    df = frame.copy()
    df[id_column] = df[id_column].astype(str)
    df = df.drop_duplicates(id_column).set_index(id_column)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.isna().mean() <= max_missing_fraction]
    df = df.fillna(df.median())
    df = df.loc[:, df.std(ddof=0) > 0]
    if prefix:
        df = df.add_prefix(prefix)
    df.index.name = "cell_line_id"
    return df.astype(np.float32)
