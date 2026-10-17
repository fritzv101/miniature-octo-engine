"""Column mappings so external tables can be loaded without renaming files by hand."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class SchemaError(ValueError):
    """Raised when an input table is missing columns a loader needs."""


@dataclass(frozen=True)
class CompoundColumns:
    compound_id: str = "compound_id"
    smiles: str = "smiles"
    name: str | None = "name"


@dataclass(frozen=True)
class BindingColumns:
    """Columns of a compound-target bioactivity table (ChEMBL-like)."""

    compound_id: str = "compound_id"
    smiles: str | None = "smiles"
    target_id: str = "target_id"
    value: str = "standard_value"
    units: str | None = "standard_units"
    assay_type: str | None = "standard_type"
    relation: str | None = "standard_relation"
    target_name: str | None = None


@dataclass(frozen=True)
class ResponseColumns:
    """Columns of a compound-cell line response table (GDSC-like).

    ``higher_is_more_sensitive`` is False for IC50/AUC-style values and True for
    sensitivity scores such as DSS. ``transform="log1p"`` compresses heavy-tailed,
    non-negative scores before modelling.
    """

    compound_id: str = "compound_id"
    cell_line_id: str = "cell_line_id"
    value: str = "ln_ic50"
    metric: str | None = None
    auc: str | None = "auc"
    higher_is_more_sensitive: bool = False
    transform: str | None = None


@dataclass(frozen=True)
class CellLineColumns:
    cell_line_id: str = "cell_line_id"
    cancer_type: str | None = "cancer_type"
    tissue: str | None = "tissue"


@dataclass(frozen=True)
class DatasetSchema:
    """Full schema bundle for one project's files."""

    compounds: CompoundColumns = field(default_factory=CompoundColumns)
    binding: BindingColumns = field(default_factory=BindingColumns)
    response: ResponseColumns = field(default_factory=ResponseColumns)
    cell_lines: CellLineColumns = field(default_factory=CellLineColumns)


GDSC_SCHEMA = DatasetSchema(
    compounds=CompoundColumns(compound_id="DRUG_ID", smiles="smiles", name="DRUG_NAME"),
    response=ResponseColumns(
        compound_id="DRUG_ID", cell_line_id="COSMIC_ID", value="LN_IC50", auc="AUC"
    ),
    cell_lines=CellLineColumns(cell_line_id="COSMIC_ID", cancer_type="TCGA_DESC", tissue="tissue"),
)

# LINCS L1000 compounds (BRD ids) x DepMap cell lines (ACH ids) with PharmacoDB DSS3
# sensitivity scores; compound table from LINCS compoundinfo, cell lines from DepMap Model.csv.
LINCS_DEPMAP_SCHEMA = DatasetSchema(
    compounds=CompoundColumns(compound_id="pert_id", smiles="smiles", name="cmap_name"),
    binding=BindingColumns(
        compound_id="pert_id",
        smiles="smiles",
        target_id="target_chembl_id",
        value="standard_value",
        units="standard_units",
        assay_type="standard_type",
        relation="standard_relation",
        target_name="target_pref_name",
    ),
    response=ResponseColumns(
        compound_id="pert_id",
        cell_line_id="cell_line",
        value="sensitivity_score",
        auc=None,
        higher_is_more_sensitive=True,
        transform="log1p",
    ),
    cell_lines=CellLineColumns(
        cell_line_id="ModelID", cancer_type="OncotreeLineage", tissue="OncotreePrimaryDisease"
    ),
)

CHEMBL_SCHEMA = DatasetSchema(
    compounds=CompoundColumns(
        compound_id="molecule_chembl_id", smiles="canonical_smiles", name="pref_name"
    ),
    binding=BindingColumns(
        compound_id="molecule_chembl_id",
        smiles="canonical_smiles",
        target_id="target_chembl_id",
        value="standard_value",
        units="standard_units",
        assay_type="standard_type",
        relation="standard_relation",
        target_name="target_pref_name",
    ),
)


def require_columns(frame: pd.DataFrame, columns: list[str], *, what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"{what} table is missing columns {missing}; available columns: {list(frame.columns)}"
        )


def optional_column(frame: pd.DataFrame, column: str | None) -> str | None:
    """Return the column name only if the caller asked for it and it exists."""
    return column if column and column in frame.columns else None
