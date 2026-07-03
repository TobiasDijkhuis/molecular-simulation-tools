"""Collection of utility functions."""

import itertools
from random import random

import numpy as np
from ase import Atoms
from scipy.signal import correlate


def get_nyquist_frequency(dt: float) -> float:
    """Get the nyquist frequency.

    Parameters
    ----------
    dt : float
        Time between samples in s

    Returns
    -------
    nyquist_frequency : float
        Nyquist frequency in Hz

    """
    return 1.0 / (2.0 * dt)


def get_autocorrelation_function(x: np.ndarray) -> np.ndarray:
    """Get the autocorrelation function of x.

    Parameters
    ----------
    x : np.ndarray
        Array of length N

    Returns
    -------
    np.ndarray
        Array of length N

    """
    return correlate(x, x)[: len(x) + 1]


def get_moving_average(array: np.ndarray, window_size: int) -> np.ndarray:
    """Get the moving average of an array with a certain window size.

    Taken from https://stackoverflow.com/a/14314054.

    Parameters
    ----------
    array : np.ndarray
        array to be smoothed of length ``N``
    window_size : int
        window size

    Returns
    -------
    np.ndarray
        Array of length ``N - window_size + 1``

    """
    array = np.cumsum(array, dtype=float)
    array[window_size:] = array[window_size:] - array[:-window_size]  # noqa: PLR6104
    return array[window_size - 1 :] / window_size


def turn_grid_into_position_vectors(
    grid_matrices: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Turn a grid created by :func:`numpy.meshgrid` into position vectors.

    Taken from https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points

    Parameters
    ----------
    grid_matrices : tuple[np.ndarray, ...]
        Tuple of N grid matrices with M points.

    Returns
    -------
    np.ndarray
        MxN numpy array of the grid positions in N dimensions.

    """
    return np.vstack(list(map(np.ravel, grid_matrices))).T


def correct_distance_for_pbc(distance: np.ndarray, box_length: float) -> np.ndarray:
    """Correct a distance for periodic boundary conditions.

    Parameters
    ----------
    distance : np.ndarray
        Array of distances
    box_length : float
        Length of the periodic box along the dimension of `distance`

    Returns
    -------
    distance : np.ndarray
        Corrected distance

    """
    distance[distance > box_length * 0.5] -= box_length
    distance[distance <= -box_length * 0.5] += box_length
    return distance


def convert_spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates.

    See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

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
    return np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )


def convert_cartesian_to_spherical(position: np.ndarray) -> tuple[float, float, float]:
    """Convert cartesian coordinates to spherical coordinates.

    See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Parameters
    ----------
    position : np.ndarray
        Cartesian coordinate vector

    Returns
    -------
    tuple[float, float, float]
        Tuple containing (r, theta, phi)

    Raises
    ------
    ValueError
        If `position` is not a 3D vector, i.e. does not have shape ``(3,)``.

    """
    if not position.shape == (3,):
        raise ValueError()
    r: float = np.linalg.norm(position)  # type: ignore[assignment, ty:invalid-assignment]
    theta = np.arccos(position[2] / r)
    phi = np.atan2(position[1], position[0])
    return r, theta, phi


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


def get_random_unit_vector(n: int = 1) -> np.ndarray:
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
    rotation_axis = get_random_unit_vector()
    rotation_angle = 2 * np.pi * random()
    return rotation_axis, rotation_angle


def check_same_number_of_atoms(frames: list[Atoms]) -> None:
    """Check that all frames have the same number of atoms.

    Parameters
    ----------
    frames : list[Atoms]
        Frames of the trajectory or calculations

    Raises
    ------
    ValueError
        If not all frames have the same number of atoms as the first frame.

    """
    n_atoms = len(frames[0])
    if not all(len(frame) == n_atoms for frame in frames[1:]):
        raise ValueError


def get_permutations_exchange_identical_atoms(
    atoms: Atoms,
    indices: list[int] | None = None,
) -> list[list[int]]:
    """Get all possible permutations of `indices` that exchange identical atoms.

    Parameters
    ----------
    atoms : Atoms
        Atoms to permute.
    indices : list[int] | None
        Indices of atoms to take into account in the permuting.
        If None, do all atoms. Default = None.

    Returns
    -------
    list[list[int]]
        List of possible correct permutations that exchange identical atoms.

    """
    if indices is None:
        indices = list(range(len(atoms)))
    permutations = [
        list(permutation) for permutation in itertools.permutations(indices)
    ]
    return [
        permutation
        for permutation in permutations
        if np.all(atoms.numbers[permutation] == atoms.numbers[indices])
    ]
