import pytest
from ase import Atoms

from molecular_simulation_tools.neb import get_images_for_neb


def test_get_images_for_neb_with_too_few_images_raises():
    initial_atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
    final_atoms = Atoms(symbols="H", positions=[[0, 0, 1]])
    n_images = 2
    with pytest.raises(ValueError):
        get_images_for_neb(initial_atoms, final_atoms, n_images)


def test_get_images_for_neb():
    initial_atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
    final_atoms = Atoms(symbols="H", positions=[[0, 0, 1]])
    n_images = 5
    neb_images = get_images_for_neb(initial_atoms, final_atoms, n_images)

    assert len(neb_images) == n_images
    assert all(neb_image == initial_atoms for neb_image in neb_images[:-1])
    assert neb_images[-1] == final_atoms

    assert id(neb_images[0]) != id(initial_atoms)
    assert id(neb_images[-1]) != id(final_atoms)
