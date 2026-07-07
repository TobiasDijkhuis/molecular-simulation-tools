import numpy as np
import pytest
from ase import Atoms

from molecular_simulation_tools.utils import (
    combine_overlapping_sets,
    convert_cartesian_to_spherical,
    convert_spherical_to_cartesian,
    get_permutations_exchange_identical_atoms,
    icosahedron_unit_sphere,
    project_on_unit_sphere,
    sample_new_point,
    turn_grid_into_position_vectors,
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


def test_icosahedron_unit_sphere():
    vertices = icosahedron_unit_sphere(level=0)

    assert np.allclose(np.linalg.norm(vertices, axis=1), 1)
    assert np.shape(vertices) == (12, 3)
    phi = 2 * np.cos(np.pi / 5)
    expected = np.array(
        [
            [0, phi, 1],
            [0, -phi, 1],
            [0, phi, -1],
            [0, -phi, -1],
            [1, 0, phi],
            [-1, 0, phi],
            [1, 0, -phi],
            [-1, 0, -phi],
            [phi, 1, 0],
            [-phi, 1, 0],
            [phi, -1, 0],
            [-phi, -1, 0],
        ]
    )
    expected /= np.linalg.norm(expected, axis=1)[:, np.newaxis]
    assert np.allclose(vertices, expected)


icosahedron_data = [
    (0, 12),
    (1, 42),
    (2, 162),
]


@pytest.mark.parametrize("level, expected_number_of_vertices", icosahedron_data)
def test_icosahedron_unit_sphere_shapes(level, expected_number_of_vertices):
    vertices = icosahedron_unit_sphere(level=level)
    assert np.shape(np.unique(vertices, axis=0)) == np.shape(vertices)
    assert np.shape(vertices)[0] == expected_number_of_vertices


sample_data = [
    (np.array([[0.0, 0.0, 0.0]]), 0.5, 1e-12, 1e-5),
    (np.array([[1.0, 0.0, 0.5], [0.0, 4.0, 2.0]]), 2, 1e-12, 1e-5),
]


@pytest.mark.parametrize("initial_points, minimum_distance, atol, rtol", sample_data)
def test_sample_new_point(initial_points, minimum_distance, atol, rtol):
    for _try_idx in range(10):
        new_point = sample_new_point(
            initial_points, minimum_distance, 0.2, rtol=rtol, atol=atol, n=1
        )
        assert np.all(
            np.linalg.norm(initial_points - new_point, axis=1)
            >= minimum_distance - atol
        )


@pytest.mark.parametrize("_initial_points, minimum_distance, atol, rtol", sample_data)
def test_sample_multiple_new_point(_initial_points, minimum_distance, atol, rtol):
    initial_point = np.array([[0, 0, 0]])
    npoints = 10
    for _try_idx in range(10):
        points = sample_new_point(
            initial_point, minimum_distance, 0.2, rtol=rtol, atol=atol, n=npoints
        )
        assert np.shape(points) == (10, 3)
        for point_idx in range(npoints):
            assert np.all(
                np.linalg.norm(points[point_idx, :] - points, axis=1)
                >= minimum_distance - atol,
                where=[idx != point_idx for idx in range(npoints)],
            )
