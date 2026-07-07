import numpy as np
import pytest
from ase import Atoms

from molecular_simulation_tools.utils import (
    convert_cartesian_to_spherical,
    convert_spherical_to_cartesian,
    get_permutations_exchange_identical_atoms,
    project_on_unit_sphere,
    turn_grid_into_position_vectors,
    combine_overlapping_sets,
)


def test_project_on_unit_sphere():
    for dimensions in range(2, 8):
        random_vec = np.random.random(dimensions)
        projected_vec = project_on_unit_sphere(random_vec)
        length = np.linalg.norm(projected_vec)
        assert np.allclose(length, 1.0)


spherical_data = [
    ((1.0, 0.0, 0.0), np.array([0.0, 0.0, 1.0])),
    ((2.0, 0.0, 0.0), np.array([0.0, 0.0, 2.0])),
    ((1.0, np.pi, 0.0), np.array([0.0, 0.0, -1.0])),
]


@pytest.mark.parametrize("spherical, cartesian", spherical_data)
def test_spherical_and_cartesian_conversion(spherical, cartesian):
    r, theta, phi = spherical
    assert np.allclose(convert_spherical_to_cartesian(r, theta, phi), cartesian)
    assert np.allclose(convert_cartesian_to_spherical(cartesian), spherical)


@pytest.mark.parametrize("spherical, cartesian", spherical_data)
def test_conserves_spherical_cartesian_round_trip(spherical, cartesian):
    assert np.allclose(
        convert_spherical_to_cartesian(*convert_cartesian_to_spherical(cartesian)),
        cartesian,
    )
    assert np.allclose(
        convert_cartesian_to_spherical(convert_spherical_to_cartesian(*spherical)),
        spherical,
    )


position_vector_data = [
    (
        (
            np.array([[0.25, 0.25], [0.75, 0.75]]),
            np.array([[0.25, 0.75], [0.25, 0.75]]),
        ),
        np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]),
    ),
]


@pytest.mark.parametrize("grid_tuple, expected_output", position_vector_data)
def test_turn_grid_into_position_vectors(grid_tuple, expected_output):
    assert np.allclose(turn_grid_into_position_vectors(grid_tuple), expected_output)


get_permutations_exchange_identical_atoms_data = [
    (Atoms("H2"), [0, 1], {(0, 1), (1, 0)}),
    (Atoms("H2O"), [0, 1], {(0, 1), (1, 0)}),
    (Atoms("H2O"), [0, 1, 2], {(0, 1, 2), (1, 0, 2)}),
    (Atoms("H2O"), None, {(0, 1, 2), (1, 0, 2)}),
    (
        Atoms("CH3"),
        [0, 1, 2, 3],
        {
            (0, 1, 2, 3),
            (0, 1, 3, 2),
            (0, 2, 1, 3),
            (0, 2, 3, 1),
            (0, 3, 1, 2),
            (0, 3, 2, 1),
        },
    ),
]


@pytest.mark.parametrize(
    "atoms, indices, expected_permutations",
    get_permutations_exchange_identical_atoms_data,
)
def test_get_permutations_exchange_identical_atoms_data(
    atoms, indices, expected_permutations
):
    permutations = get_permutations_exchange_identical_atoms(atoms, indices=indices)
    print(permutations)
    assert len(permutations) == len(expected_permutations)
    assert all(
        tuple(permutation) in expected_permutations for permutation in permutations
    )


overlapping_sets_data = [
    ([{1, 2, 3}, {4, 5}], [{1, 2, 3}, {4, 5}]),
    ([{1, 2, 3}, {3, 5}], [{1, 2, 3, 5}]),
    ([{1, 2, 3}, {4, 5}, {1, 5}], [{1, 2, 3, 4, 5}]),
    ([{1}, {2}, {3}], [{1}, {2}, {3}]),
    ([{1, 2}, {2, 3}, {4, 5}, {5, 6}], [{1, 2, 3}, {4, 5, 6}]),
    ([{1, 2}, {1, 2}, {4, 5}, {5, 6}], [{1, 2}, {4, 5, 6}]),
]


@pytest.mark.parametrize("list_of_sets, expected_sets", overlapping_sets_data)
def test_combine_overlapping_sets(list_of_sets, expected_sets):
    combined_sets = combine_overlapping_sets(list_of_sets)
    assert expected_sets == combined_sets
