import numpy as np
import pytest

from molecular_simulation_tools.utils import (
    convert_cartesian_to_spherical,
    convert_spherical_to_cartesian,
    project_on_unit_sphere,
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
