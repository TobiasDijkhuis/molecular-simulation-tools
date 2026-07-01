import numpy as np
import pytest
from ase import Atoms

from molecular_simulation_tools.connectivity import (
    identify_molecules,
)

identify_molecules_data = [
    (
        Atoms(symbols="OH2", positions=np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]])),
        {"H": 0.5, "O": 0.5},
        [np.array([0, 1, 2])],
    ),
    (
        Atoms(symbols="OH2", positions=np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]])),
        {"H": 0.2, "O": 0.2},
        [np.array([0]), np.array([1]), np.array([2])],
    ),
]


@pytest.mark.parametrize("atoms, cutoffs, expected_molecules", identify_molecules_data)
def test_identify_molecules(atoms, cutoffs, expected_molecules):
    molecules = identify_molecules(atoms, cutoffs=cutoffs)
    for idx, molecule in enumerate(molecules):
        assert np.all(molecule == expected_molecules[idx])
