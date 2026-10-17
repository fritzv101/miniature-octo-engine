"""Download and assemble public reference data for the LINCS/DepMap dataset.

The response table (LINCS BRD compound ids x DepMap ACH cell lines) has no
structures, targets or cell-line annotations; these helpers pull them from:

* LINCS 2020 ``compoundinfo_beta.txt``: BRD id -> SMILES, InChIKey, annotated targets/MOA
* DepMap public release on figshare: ``Model.csv`` (lineage/disease), hotspot
  mutation matrix, protein-coding expression (TPM log1p)
* ChEMBL REST API: InChIKey -> ChEMBL id -> human single-protein binding activities

Requires the ``fetch`` extra (``requests``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from oncomap.chem import try_standardize_smiles

logger = logging.getLogger(__name__)

LINCS_COMPOUNDINFO_URL = (
    "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt"
)
DEPMAP_FIGSHARE_ARTICLE = 27993248  # DepMap 24Q4 Public
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_ACTIVITY_FIELDS = (
    "molecule_chembl_id,target_chembl_id,target_pref_name,target_organism,standard_type,"
    "standard_relation,standard_value,standard_units,pchembl_value,assay_type,"
    "data_validity_comment"
)


def _session() -> Any:
    import requests

    return requests.Session()


def _get_json(session: Any, url: str, params: dict[str, Any] | None = None) -> Any:
    for attempt in range(5):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # network hiccups and 5xx from public APIs
            logger.warning("GET %s failed (%s), retry %d", url, exc, attempt + 1)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def download(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return dest
    with _session().get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def lincs_compounds(pert_ids: set[str], cache_dir: Path) -> pd.DataFrame:
    """BRD id -> standardized SMILES, InChIKey, annotated targets and MOA."""
    path = download(LINCS_COMPOUNDINFO_URL, cache_dir / "compoundinfo_beta.txt")
    info = pd.read_csv(path, sep="\t")
    info = info[info["pert_id"].isin(pert_ids)].copy()
    info["smiles"] = info["canonical_smiles"].map(try_standardize_smiles)

    def _join(values: pd.Series) -> str:
        return ";".join(sorted({v for v in values.dropna() if v and v != '""'}))

    annotations = info.groupby("pert_id").agg(
        annotated_targets=("target", _join), moa=("moa", _join)
    )
    first = info.dropna(subset=["smiles"]).drop_duplicates("pert_id")
    out = first[["pert_id", "cmap_name", "smiles", "inchi_key"]].merge(
        annotations, left_on="pert_id", right_index=True
    )
    logger.info("LINCS: %d of %d compounds have a structure", len(out), len(pert_ids))
    return out.reset_index(drop=True)


def depmap_file(name: str, cache_dir: Path, article: int = DEPMAP_FIGSHARE_ARTICLE) -> Path:
    """Download one file of a DepMap figshare release by name (e.g. ``Model.csv``)."""
    files = _get_json(
        _session(), f"https://api.figshare.com/v2/articles/{article}/files", {"page_size": 1000}
    )
    match = next((f for f in files if f["name"] == name), None)
    if match is None:
        raise FileNotFoundError(f"{name} not in figshare article {article}")
    return download(match["download_url"], cache_dir / name)


def read_depmap_expression(path: str | Path) -> pd.DataFrame:
    """Cell line x gene TPM matrix from either DepMap expression layout.

    Older releases are indexed by ModelID; newer ones have one row per sequencing
    run with ``ModelID`` and ``IsDefaultEntryForModel`` columns.
    """
    expr = pd.read_csv(path, index_col=0)
    if "ModelID" in expr.columns:
        if "IsDefaultEntryForModel" in expr.columns:
            expr = expr[expr["IsDefaultEntryForModel"] == "Yes"]
        expr = expr.set_index("ModelID")
    expr = expr.select_dtypes("number")
    return expr[~expr.index.duplicated()]


def depmap_cell_features(
    cell_line_ids: set[str],
    cache_dir: Path,
    *,
    expression_path: str | Path | None = None,
    n_genes: int = 1000,
    min_mut_freq: float = 0.01,
) -> pd.DataFrame:
    """Top-variance expression genes plus frequent hotspot mutations, per cell line.

    Mutation columns are 0 for profiled lines without a hotspot; expression stays
    NaN for lines without RNA-seq so downstream imputation can see it is missing.
    """
    expr_path = expression_path or depmap_file(
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv", cache_dir
    )
    expr = read_depmap_expression(expr_path)
    expr = expr.loc[expr.index.intersection(sorted(cell_line_ids))]
    top = expr[expr.var().sort_values(ascending=False).index[:n_genes]]
    top.columns = ["expr_" + c.split(" (")[0] for c in top.columns]

    mut = pd.read_csv(depmap_file("OmicsSomaticMutationsMatrixHotspot.csv", cache_dir), index_col=0)
    mut = (mut > 0).astype("int8")
    mut = mut.loc[mut.index.intersection(sorted(cell_line_ids))]
    mut = mut.loc[:, mut.mean() >= min_mut_freq]
    mut.columns = ["mut_" + c.split(" (")[0] for c in mut.columns]

    features = top.join(mut, how="outer").reindex(sorted(cell_line_ids)).dropna(how="all")
    features[mut.columns] = features[mut.columns].fillna(0)
    features.index.name = "cell_line_id"
    return features.astype("float32")


def chembl_binding(inchi_keys: list[str], *, batch: int = 20) -> pd.DataFrame:
    """Human single-protein binding activities (with pChEMBL) for the given InChIKeys."""
    session = _session()
    keys = [k for k in dict.fromkeys(inchi_keys) if isinstance(k, str) and len(k) == 27]
    mols = []
    for i in range(0, len(keys), 50):
        data = _get_json(
            session,
            f"{CHEMBL_API}/molecule.json",
            {
                "molecule_structures__standard_inchi_key__in": ",".join(keys[i : i + 50]),
                "limit": 1000,
                "only": "molecule_chembl_id,molecule_structures",
            },
        )
        for m in data["molecules"]:
            structures = m.get("molecule_structures") or {}
            mols.append(
                {
                    "molecule_chembl_id": m["molecule_chembl_id"],
                    "inchi_key": structures.get("standard_inchi_key"),
                }
            )
    id_map = pd.DataFrame(mols, columns=["molecule_chembl_id", "inchi_key"]).drop_duplicates()
    chembl_ids = id_map["molecule_chembl_id"].tolist()

    activities: list[dict[str, Any]] = []
    for i in range(0, len(chembl_ids), batch):
        offset = 0
        while True:
            data = _get_json(
                session,
                f"{CHEMBL_API}/activity.json",
                {
                    "molecule_chembl_id__in": ",".join(chembl_ids[i : i + batch]),
                    "pchembl_value__isnull": "false",
                    "target_organism": "Homo sapiens",
                    "assay_type": "B",
                    "limit": 1000,
                    "offset": offset,
                    "only": CHEMBL_ACTIVITY_FIELDS,
                },
            )
            activities.extend(data["activities"])
            offset += 1000
            if not data["page_meta"]["next"]:
                break
        logger.info("ChEMBL: %d activities after %d molecules", len(activities), i + batch)
    acts = pd.DataFrame(activities)
    if acts.empty:
        return acts
    return acts.merge(id_map, on="molecule_chembl_id", how="left")


def build_lincs_depmap_dataset(
    response_path: str | Path,
    out_dir: str | Path,
    *,
    expression_path: str | Path | None = None,
) -> dict[str, Path]:
    """Assemble compounds/binding/cell_lines/cell_features tables next to a response file.

    Outputs are ready for ``oncomap train --schema lincs_depmap``.
    """
    out = Path(out_dir)
    cache = out / "cache"
    response = pd.read_csv(response_path, usecols=["pert_id", "cell_line"])
    pert_ids, cell_ids = set(response["pert_id"]), set(response["cell_line"])

    compounds = lincs_compounds(pert_ids, cache)
    acts = chembl_binding(compounds["inchi_key"].dropna().tolist())
    binding = acts.merge(compounds[["pert_id", "inchi_key"]], on="inchi_key", how="inner")
    binding = binding[binding["data_validity_comment"].isna()]
    # pChEMBL is already -log10(M) and harmonized across IC50/Ki/Kd/EC50.
    binding = binding.assign(
        standard_type="pChEMBL", standard_value=binding["pchembl_value"], standard_relation="="
    )
    model = pd.read_csv(depmap_file("Model.csv", cache))
    cell_lines = model[model["ModelID"].isin(cell_ids)]
    features = depmap_cell_features(cell_ids, cache, expression_path=expression_path)

    paths = {
        "compounds": out / "compounds.csv",
        "binding": out / "binding.csv",
        "cell_lines": out / "cell_lines.csv",
        "cell_features": out / "cell_features.csv",
    }
    compounds.to_csv(paths["compounds"], index=False)
    binding.to_csv(paths["binding"], index=False)
    cell_lines.to_csv(paths["cell_lines"], index=False)
    features.to_csv(paths["cell_features"])
    return paths
