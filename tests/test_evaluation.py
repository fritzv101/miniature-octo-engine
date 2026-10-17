import numpy as np

from oncomap.evaluation import group_split, regression_metrics, scaffold_split


def test_scaffold_split_shares_no_scaffold():
    smiles = ["Cc1ccc2ncnc(NC)c2c1", "Clc1ccc2ncnc(NCC)c2c1", "CCO", "CCCO", "c1ccccc1"]
    train, test = scaffold_split(smiles, test_fraction=0.4, seed=1)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(smiles)
    from oncomap.chem import murcko_scaffold

    train_scaffolds = {murcko_scaffold(smiles[i]) for i in train}
    test_scaffolds = {murcko_scaffold(smiles[i]) for i in test}
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_group_split_keeps_groups_whole():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    train, test = group_split(groups, test_fraction=0.25, seed=0)
    test_groups = {groups[i] for i in test}
    assert all(groups[i] not in test_groups for i in train)


def test_regression_metrics_perfect_and_empty():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0 and m["r2"] == 1 and m["pearson"] > 0.99
    assert regression_metrics(np.array([]), np.array([]))["n"] == 0
