import numpy as np
import pytest

from oncomap.chem import (
    FeaturizerConfig,
    InvalidSmilesError,
    MoleculeFeaturizer,
    descriptors,
    lipinski_violations,
    murcko_scaffold,
    standardize_smiles,
    try_standardize_smiles,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_standardize_is_idempotent_and_order_invariant():
    a = standardize_smiles(ASPIRIN)
    assert standardize_smiles(a) == a
    assert standardize_smiles("OC(=O)c1ccccc1OC(C)=O") == a


def test_standardize_strips_salt_and_neutralizes():
    assert standardize_smiles("CC(=O)[O-].[Na+]") == standardize_smiles("CC(=O)O")


def test_invalid_smiles_raises_and_soft_variant_returns_none():
    with pytest.raises(InvalidSmilesError):
        standardize_smiles("not_a_molecule((")
    assert try_standardize_smiles("not_a_molecule((") is None
    assert try_standardize_smiles("") is None


def test_featurizer_shape_and_determinism():
    f = MoleculeFeaturizer(FeaturizerConfig(fingerprint_bits=256))
    x = f.featurize(ASPIRIN)
    assert x.shape == (f.n_features,) == (len(f.feature_names),)
    assert np.array_equal(x, f.featurize(ASPIRIN))
    assert f.featurize_many([]).shape == (0, f.n_features)


def test_different_molecules_have_different_fingerprints():
    f = MoleculeFeaturizer(FeaturizerConfig(fingerprint_bits=512))
    assert not np.array_equal(f.featurize(ASPIRIN), f.featurize("c1ccccc1"))


def test_descriptors_match_known_values():
    d = descriptors(ASPIRIN)
    assert d["mol_weight"] == pytest.approx(180.16, abs=0.1)
    assert d["aromatic_rings"] == 1
    assert lipinski_violations(ASPIRIN) == 0
    assert lipinski_violations("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC") >= 1


def test_scaffold_groups_analogues_together():
    core = murcko_scaffold("Cc1ccc2ncnc(NC)c2c1")
    assert core == murcko_scaffold("Clc1ccc2ncnc(NCC)c2c1")
    assert murcko_scaffold("CCCC") == ""
