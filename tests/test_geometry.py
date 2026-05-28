import numpy as np
import pytest
from ase import Atoms

from molecular_simulation_tools.geometry import (
    calculate_rmsd,
    construct_grid_in_cell,
    discretize_cell_length,
    find_min_height_for_distance,
    icosahedron_unit_sphere,
    sample_new_point,
)

discretization_data = [
    (1.0, 2, np.array([0.25, 0.75])),
    (5.0, 2, np.array([1.25, 3.75])),
    (1.0, 3, np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])),
    (1.0, 4, np.array([1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0])),
]


@pytest.mark.parametrize("length, num, expected_output", discretization_data)
def test_discretize_cell_length(length, num, expected_output):
    output = discretize_cell_length(length, num)

    if num % 2 == 1:
        assert output[num // 2] == length / 2

    # Make sure that the spacing is as far to the right as it is to the left.
    assert np.allclose(length - output[-1], output[0])
    assert np.allclose(output, expected_output)


grid_data = [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        2,
        (
            np.array([[0.25, 0.25], [0.75, 0.75]]),
            np.array([[0.25, 0.75], [0.25, 0.75]]),
        ),
    ),
    (
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]]),
        2,
        (
            np.array([[0.25, 0.25], [0.75, 0.75]]),
            np.array([[0.5, 1.5], [0.5, 1.5]]),
        ),
    ),
]


@pytest.mark.parametrize("cell, num, expected_output", grid_data)
def test_construct_grid_in_cell(cell, num, expected_output):
    output = construct_grid_in_cell(cell, num)
    assert np.allclose(output[0], expected_output[0])
    assert np.allclose(output[1], expected_output[1])


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
        for point_idx in range(npoints):
            assert np.all(
                np.linalg.norm(points[point_idx, :] - points, axis=1)
                >= minimum_distance - atol,
                where=[idx != point_idx for idx in range(npoints)],
            )


def test_calculate_rmsd_with_different_symbols_raises():
    with pytest.raises(ValueError):
        calculate_rmsd(
            Atoms(symbols="H", positions=[[0, 0, 0]]),
            Atoms(symbols="O", positions=[[0, 0, 0]]),
            None,
        )


rmsd_data = [
    (Atoms("H", positions=[[0, 0, 0]]), Atoms("H", positions=[[0, 0, 0]]), True, 0),
    (Atoms("H", positions=[[0, 0, 0]]), Atoms("H", positions=[[1, 0, 0]]), True, 1.0),
    (
        Atoms("H", positions=[[0, 0, 0]], cell=[1.0, 1.0, 1.0], pbc=True),
        Atoms("H", positions=[[1, 0, 0]], cell=[1.0, 1.0, 1.0], pbc=True),
        True,
        0.0,
    ),
    (
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]),
        Atoms("H2", positions=[[0, 0, 1], [0, 0, 0]]),
        True,
        0.0,
    ),
    (
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]),
        Atoms("H2", positions=[[0, 0, 1], [0, 0, 0]]),
        False,
        1.0,
    ),
    (
        Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        Atoms("H2O", positions=[[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        True,
        np.sqrt(2 / 3),
    ),
]


@pytest.mark.parametrize("atoms_a, atoms_b, permute, expected_rmsd", rmsd_data)
def test_calculate_rmsd(atoms_a, atoms_b, permute, expected_rmsd):
    assert calculate_rmsd(atoms_a, atoms_b, None, permute)[0] == expected_rmsd


min_height_for_distance_data = [
    # It needs to be at [1, 0, 1] (which is sqrt(2) distance from [0, 0, 0])
    (1.0, 0.0, np.array([[0, 0, 0]]), np.sqrt(2), None, 1),
    # Still needs to be at [1, 0, 1], because that is also sqrt(2) away from [0, 1, 1]
    (1.0, 0.0, np.array([[0, 0, 0], [0, 1, 1]]), np.sqrt(2), None, 1),
    # Due to PBCs, this one needs to be at z = 1 (it is directly above the point)
    (1.0, 0.0, np.array([[0, 0, 0]]), 1, [1, 1, 1], 1),
]


@pytest.mark.parametrize(
    "x, y, point_coordinates, distance, box_size, expected_height",
    min_height_for_distance_data,
)
def test_find_min_height_for_distance(
    x, y, point_coordinates, distance, box_size, expected_height
):
    assert np.allclose(
        find_min_height_for_distance(
            x, y, point_coordinates, distance, box_size=box_size
        ),
        expected_height,
    )


def test_find_min_height_for_distance_raises_if_none_close():
    with pytest.raises(ValueError):
        find_min_height_for_distance(0, 0, np.array([[1, 0, 0]]), 0.5)


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
