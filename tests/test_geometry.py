import numpy as np
import pytest
from ase import Atoms
from ase.constraints.fix_atoms import FixAtoms

from molecular_simulation_tools.geometry import (
    calculate_rmsd,
    construct_grid_in_cell,
    cut_out_atoms_within_radius,
    cut_out_trajectory_within_radius,
    discretize_cell_length,
    find_min_height_for_distance,
    get_constraint_outside_radius,
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


cutout_data = [
    (
        Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        np.array([0, 0, 0]),
        0.5,
        np.array([[0, 0, 0]]),
    ),
    (
        Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        np.array([0, 0, 0]),
        1,
        np.array([[0, 0, 0], [0, 0, 1]]),
    ),
    (
        Atoms(
            "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[3, 3, 3], pbc=True
        ),
        np.array([0, 0, 0]),
        1,
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]]),
    ),
]


@pytest.mark.parametrize("atoms, center, radius, expected_positions", cutout_data)
def test_cut_out_atoms_within_radius(atoms, center, radius, expected_positions):
    cutout_atoms = cut_out_atoms_within_radius(
        atoms, center, radius, keep_molecules_intact=False
    )
    assert np.all(np.linalg.norm(cutout_atoms.positions - center, axis=1) <= radius)
    assert np.shape(expected_positions)[0] == np.shape(cutout_atoms.positions)[0]
    assert np.allclose(expected_positions, cutout_atoms.positions)


cutout_from_trajectory_data = [
    (
        [Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])] * 2,
        [np.array([0, 0, 0]), np.array([0, 0, 1])],
        0.5,
        [np.array([[0, 0, 0], [0, 0, 1]]), np.array([[0, 0, -1], [0, 0, 0]])],
    ),
    (
        [Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])] * 3,
        [np.array([0, 0, 0]), np.array([0, 0, 0.5]), np.array([0, 0, 1])],
        0.25,
        [
            np.array([[0, 0, 0], [0, 0, 1]]),
            np.array([[0, 0, -0.5], [0, 0, 0.5]]),
            np.array([[0, 0, -1], [0, 0, 0]]),
        ],
    ),
]


@pytest.mark.parametrize(
    "frames, centers, radius, expected_positions", cutout_from_trajectory_data
)
def test_cut_out_trajectory_within_radius(frames, centers, radius, expected_positions):
    trajectory_within_radius = cut_out_trajectory_within_radius(
        frames, centers, radius, keep_molecules_intact=False
    )

    for frame_idx in range(len(frames)):
        assert np.all(
            expected_positions[frame_idx]
            == trajectory_within_radius[frame_idx].positions
        )


constraint_data = [
    (
        Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        np.array([0, 0, 0]),
        0.5,
        FixAtoms(indices=[1, 2]),
    ),
    (
        Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        np.array([0, 0, 0]),
        1,
        FixAtoms(indices=[2]),
    ),
    (
        Atoms(
            "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[3, 3, 3], pbc=True
        ),
        np.array([0, 0, 0]),
        1,
        FixAtoms(indices=[]),
    ),
]


@pytest.mark.parametrize("atoms, center, radius, expected_constraint", constraint_data)
def test_get_constraint_outside_radius(atoms, center, radius, expected_constraint):
    assert all(
        get_constraint_outside_radius(atoms, center, radius).get_indices()
        == expected_constraint.get_indices()
    )
