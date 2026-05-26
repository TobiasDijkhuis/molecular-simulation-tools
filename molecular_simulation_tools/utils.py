"""Collection of utility functions."""

from random import random

import numpy as np


def convert_spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    r : float
        Radial distance
    theta : float
        Polar angle in radians
    phi : float
        Azimuthal angle in radians

    Returns
    -------
    np.ndarray
        numpy array containing (x, y, z)

    """
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    return np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )


def project_on_unit_sphere(vector: np.ndarray) -> np.ndarray:
    """Project a vector on the unit sphere, by dividing it by its length.

    Parameters
    ----------
    vector : np.ndarray
        N-dimensional vector

    Returns
    -------
    np.ndarray
        Vector projected on N-dimensional unit sphere

    """
    return vector / np.linalg.norm(vector)


def random_on_unit_sphere(n: int = 1) -> np.ndarray:
    """Get a random vector on the unit sphere.

    Uses method described in https://mathworld.wolfram.com/SpherePointPicking.html

    Parameters
    ----------
    n : int
        Number of vectors to generate. Default = 1

    Returns
    -------
    vectors : np.ndarray
        Numpy matrix of shape Nx3, or just vector of shape (3,) if
        `n` is 1.

    """
    vectors = np.empty(shape=(n, 3))
    for i in range(n):
        u1 = random()
        u2 = random()

        theta = 2.0 * np.pi * u2
        phi = np.acos(2.0 * u1 - 1.0)
        vectors[i, :] = convert_spherical_to_cartesian(1.0, theta, phi)
    if n == 1:
        return vectors[0, :]
    return vectors


def sample_random_rotation() -> tuple[np.ndarray, float]:
    """Get a random rotation axis and rotation angle.

    Uses method described in https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices

    Returns
    -------
    rotation_axis : np.ndarray
        3-dimensional rotation axis.
    rotation_angle : float
        Rotation angle in radians.

    """
    rotation_axis = random_on_unit_sphere()
    rotation_angle = 2 * np.pi * random()
    return rotation_axis, rotation_angle
