import numpy as np
import pytest

from molecular_simulation_tools.geometry import (
    construct_grid_in_cell,
    discretize_cell_length,
    turn_grid_into_position_vectors,
)

discretization_data = [
    (1.0, 2, np.array([0.25, 0.75])),
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
