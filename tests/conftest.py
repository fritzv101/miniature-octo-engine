import pytest

from oncomap.data.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def dataset():
    return make_synthetic_dataset(n_targets=6, n_cell_lines=20, seed=3)
