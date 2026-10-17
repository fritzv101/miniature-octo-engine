"""Synthetic oncology dataset with a known compound -> target -> cell line chain.

It exists so the pipeline can be run, tested and demoed without licensed data.
The generative story mirrors the biology the models assume:

1. Compounds are combinatorial analogues on drug-like cores.
2. Each target has a hidden structure-activity function over the compound's
   fingerprint, giving a pActivity per (compound, target).
3. Each cell line depends on a few targets (oncogene addiction); dependency is
   partly visible through omics features (mutation/expression flags).
4. A cell line responds to a compound when the compound potently binds the
   targets that cell line depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oncomap.chem import FeaturizerConfig, MoleculeFeaturizer, standardize_smiles

_CORES = (
    "c1ccc2ncnc(N{R1})c2c1{R2}",  # quinazoline (EGFR-like)
    "c1cc({R1})ccc1NC(=O)c1ccc({R2})cc1",  # benzanilide (type II kinase-like)
    "c1cnc(N{R1})nc1-c1ccc({R2})cc1",  # aminopyrimidine
    "O=C(N{R1})c1cc2ccccc2n1C{R2}",  # indole carboxamide
    "c1ccc(-c2nc3ccccc3[nH]2)cc1{R1}.{R2}",  # benzimidazole; R2 dropped by salt strip
    "C1CN(C(=O)c2ccc({R1})cc2)CCN1{R2}",  # piperazine amide
)

_SUBSTITUENTS = (
    "C",
    "CC",
    "OC",
    "F",
    "Cl",
    "C(F)(F)F",
    "C#N",
    "N7CCOCC7",
    "N7CCN(C)CC7",
    "c7ccncc7",
    "C(=O)N",
    "S(=O)(=O)C",
)

_CANCER_TYPES = ("lung", "breast", "colorectal", "melanoma", "leukemia", "glioma")


@dataclass
class SyntheticDataset:
    compounds: pd.DataFrame
    binding: pd.DataFrame
    response: pd.DataFrame
    cell_lines: pd.DataFrame
    cell_features: pd.DataFrame
    true_affinity: pd.DataFrame  # complete compound x target matrix, for evaluation


def enumerate_compounds() -> list[str]:
    seen: dict[str, None] = {}
    for core in _CORES:
        for r1 in _SUBSTITUENTS:
            for r2 in _SUBSTITUENTS:
                smi = core.format(R1=r1, R2=r2)
                try:
                    seen.setdefault(standardize_smiles(smi), None)
                except ValueError:
                    continue
    return list(seen)


def make_synthetic_dataset(
    n_targets: int = 12,
    n_cell_lines: int = 60,
    binding_density: float = 0.35,
    response_density: float = 0.6,
    noise: float = 0.3,
    seed: int = 7,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    smiles = enumerate_compounds()
    n_comp = len(smiles)
    compound_ids = [f"CMPD{i:04d}" for i in range(n_comp)]

    featurizer = MoleculeFeaturizer(FeaturizerConfig(fingerprint_bits=1024, use_descriptors=False))
    fps = featurizer.featurize_many(smiles)

    target_ids = [f"TGT{i:02d}" for i in range(n_targets)]
    # Sparse SAR: each target cares about ~40 fingerprint bits.
    weights = np.zeros((fps.shape[1], n_targets), dtype=float)
    for t in range(n_targets):
        bits = rng.choice(fps.shape[1], size=40, replace=False)
        weights[bits, t] = rng.normal(0, 0.8, size=40)
    raw = fps @ weights
    raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-9)
    affinity = 6.0 + 1.2 * raw  # pActivity, mostly 4-8
    affinity = np.clip(affinity, 3.0, 10.5)
    true_aff = pd.DataFrame(affinity, index=compound_ids, columns=target_ids)

    cell_ids = [f"CL{i:03d}" for i in range(n_cell_lines)]
    cancer = rng.choice(_CANCER_TYPES, size=n_cell_lines)
    # Each cancer type leans on a couple of targets; each cell line adds its own.
    type_targets = {ct: rng.choice(n_targets, size=2, replace=False) for ct in _CANCER_TYPES}
    dependency = np.zeros((n_cell_lines, n_targets))
    for c, ct in enumerate(cancer):
        dependency[c, type_targets[ct]] += rng.uniform(0.8, 1.5, size=2)
        extra = rng.choice(n_targets, size=1)
        dependency[c, extra] += rng.uniform(0.3, 1.0)

    # Omics: a mutation flag per target (driver), plus noisy expression.
    mut = (dependency > 0.7).astype(float)
    flip = rng.random(mut.shape) < 0.1
    mut = np.abs(mut - flip)
    expr = dependency + rng.normal(0, 0.4, size=dependency.shape)
    passenger = rng.normal(0, 1, size=(n_cell_lines, 20))
    cell_features = pd.DataFrame(
        np.hstack([mut, expr, passenger]),
        index=pd.Index(cell_ids, name="cell_line_id"),
        columns=[f"mut_{t}" for t in target_ids]
        + [f"expr_{t}" for t in target_ids]
        + [f"expr_passenger{i:02d}" for i in range(20)],
    ).astype(np.float32)

    # Response: ln IC50 drops when the compound hits what the cell line needs.
    occupancy = 1 / (1 + np.exp(-(affinity - 6.5) * 1.5))  # compounds x targets
    effect = occupancy @ dependency.T  # compounds x cell lines
    baseline = rng.normal(3.0, 0.5, size=n_cell_lines)
    ln_ic50 = baseline[None, :] - 2.5 * effect + rng.normal(0, noise, size=effect.shape)

    b_mask = rng.random(affinity.shape) < binding_density
    ci, ti = np.nonzero(b_mask)
    binding = pd.DataFrame(
        {
            "compound_id": np.array(compound_ids)[ci],
            "smiles": np.array(smiles)[ci],
            "target_id": np.array(target_ids)[ti],
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": 10 ** (9 - (affinity[ci, ti] + rng.normal(0, noise, size=ci.size))),
            "standard_units": "nM",
        }
    )

    r_mask = rng.random(ln_ic50.shape) < response_density
    ci, li = np.nonzero(r_mask)
    response = pd.DataFrame(
        {
            "compound_id": np.array(compound_ids)[ci],
            "cell_line_id": np.array(cell_ids)[li],
            "ln_ic50": ln_ic50[ci, li],
        }
    )

    compounds = pd.DataFrame({"compound_id": compound_ids, "smiles": smiles, "name": compound_ids})
    cell_lines = pd.DataFrame(
        {"cancer_type": cancer, "tissue": cancer}, index=pd.Index(cell_ids, name="cell_line_id")
    )
    return SyntheticDataset(compounds, binding, response, cell_lines, cell_features, true_aff)


def write_synthetic_dataset(directory: str, **kwargs: object) -> dict[str, str]:
    """Write the demo tables as CSV, returning a mapping of table name -> path."""
    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    ds = make_synthetic_dataset(**kwargs)  # type: ignore[arg-type]
    paths = {
        "compounds": out / "compounds.csv",
        "binding": out / "binding.csv",
        "response": out / "response.csv",
        "cell_lines": out / "cell_lines.csv",
        "cell_features": out / "cell_features.csv",
    }
    ds.compounds.to_csv(paths["compounds"], index=False)
    ds.binding.to_csv(paths["binding"], index=False)
    ds.response.to_csv(paths["response"], index=False)
    ds.cell_lines.to_csv(paths["cell_lines"])
    ds.cell_features.to_csv(paths["cell_features"])
    return {k: str(v) for k, v in paths.items()}
