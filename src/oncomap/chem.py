"""Molecular standardization and featurization.

Everything downstream consumes the canonical SMILES produced here, so that the
same compound written two different ways maps onto one row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "mol_weight",
    "clogp",
    "tpsa",
    "h_donors",
    "h_acceptors",
    "rotatable_bonds",
    "rings",
    "aromatic_rings",
    "heavy_atoms",
    "fraction_csp3",
    "formal_charge",
    "heteroatom_fraction",
)


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


@dataclass(frozen=True)
class FeaturizerConfig:
    """Fingerprint settings; stored with a trained model so inference matches training."""

    fingerprint_bits: int = 2048
    fingerprint_radius: int = 2
    use_descriptors: bool = True
    use_counts: bool = False


@dataclass
class MoleculeFeaturizer:
    """Turns SMILES into a fixed-width numeric vector."""

    config: FeaturizerConfig = field(default_factory=FeaturizerConfig)

    def __post_init__(self) -> None:
        self._generator = self._make_generator()

    def _make_generator(self) -> rdFingerprintGenerator.FingerprintGenerator64:
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.config.fingerprint_radius, fpSize=self.config.fingerprint_bits
        )

    # RDKit generators are not picklable; rebuild from config after unpickling.
    def __getstate__(self) -> dict[str, object]:
        return {"config": self.config}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = state["config"]  # type: ignore[assignment]
        self._generator = self._make_generator()

    @property
    def feature_names(self) -> list[str]:
        names = [f"fp_{i}" for i in range(self.config.fingerprint_bits)]
        if self.config.use_descriptors:
            names.extend(DESCRIPTOR_NAMES)
        return names

    @property
    def n_features(self) -> int:
        return self.config.fingerprint_bits + (
            len(DESCRIPTOR_NAMES) if self.config.use_descriptors else 0
        )

    def featurize(self, smiles: str) -> np.ndarray:
        mol = to_mol(smiles)
        return self.featurize_mol(mol)

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        if self.config.use_counts:
            fp = self._generator.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        else:
            fp = self._generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        if not self.config.use_descriptors:
            return fp
        return np.concatenate([fp, _descriptor_vector(mol)])

    def featurize_many(self, smiles: list[str]) -> np.ndarray:
        if not smiles:
            return np.zeros((0, self.n_features), dtype=np.float32)
        return np.vstack([self.featurize(s) for s in smiles])


def to_mol(smiles: str) -> Chem.Mol:
    """Parse a SMILES string, raising a typed error instead of returning None."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidSmilesError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSmilesError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def standardize_smiles(smiles: str, *, strip_salts: bool = True) -> str:
    """Canonicalize a SMILES: largest fragment, neutralized, canonical tautomer-safe form.

    Salt stripping keeps the largest organic fragment, which is what bioactivity
    databases effectively measure (the counter-ion is not the ligand).
    """
    mol = to_mol(smiles)
    mol = rdMolStandardize.Cleanup(mol)
    if strip_salts:
        mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise InvalidSmilesError(f"standardization emptied molecule: {smiles!r}")
    return Chem.MolToSmiles(mol)


def try_standardize_smiles(smiles: str) -> str | None:
    """Non-raising variant for bulk cleaning of messy input tables."""
    try:
        return standardize_smiles(smiles)
    except (InvalidSmilesError, Chem.rdchem.MolSanitizeException, RuntimeError):
        return None


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold, used to build splits that do not leak chemical series."""
    mol = to_mol(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def descriptors(smiles: str) -> dict[str, float]:
    """Interpretable physicochemical properties, handy for reports and filters."""
    mol = to_mol(smiles)
    values = _descriptor_vector(mol)
    return dict(zip(DESCRIPTOR_NAMES, (float(v) for v in values), strict=True))


def lipinski_violations(smiles: str) -> int:
    d = descriptors(smiles)
    return int(
        (d["mol_weight"] > 500) + (d["clogp"] > 5) + (d["h_donors"] > 5) + (d["h_acceptors"] > 10)
    )


def _descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    heavy = mol.GetNumHeavyAtoms() or 1
    hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    return np.array(
        [
            Descriptors.MolWt(mol),  # type: ignore[attr-defined]
            Crippen.MolLogP(mol),  # type: ignore[attr-defined]
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            heavy,
            rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol),
            hetero / heavy,
        ],
        dtype=np.float32,
    )
