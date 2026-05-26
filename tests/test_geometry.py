import numpy as np
import pytest

from molecular_simulation_tools.geometry import (
    construct_grid_in_cell,
    discretize_cell_length,
    sample_new_point,
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
