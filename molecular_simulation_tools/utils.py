"""Collection of utility functions."""

import itertools
from random import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from matplotlib import patches
from scipy.signal import correlate
from scipy.spatial import ConvexHull
from scipy.stats import norm
from ase.calculators.tip4p import angleHOH, rOH


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
    vectors = norm.rvs(size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1)

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


def combine_overlapping_sets(list_of_sets: list[set[Any]]) -> list[set[Any]]:
    """Combine sets that have overlapping elements.

    Parameters
    ----------
    list_of_sets : list[set[Any]]
        List of sets with potentially overlapping elements.

    Returns
    -------
    list[set[Any]]
        List of sets, without overlapping elements.

    Examples
    --------
    >>> combine_overlapping_sets([{1,2,3},{3,4}])
    [{1, 2, 3, 4}]
    >>> combine_overlapping_sets([{1,2},{3,4}])
    [{1, 2}, {3, 4}]

    """
    remaining_sets = list_of_sets.copy()
    non_overlapping_sets = []

    while remaining_sets:
        merged = remaining_sets.pop()

        changed = True
        while changed:
            changed = False
            still_remaining = []

            for s in remaining_sets:
                if not merged.isdisjoint(s):
                    merged.update(s)
                    changed = True
                else:
                    still_remaining.append(s)

            remaining_sets = still_remaining

        non_overlapping_sets.append(merged)

    return non_overlapping_sets[::-1]


def sample_new_point(
    points: np.ndarray,
    minimum_distance: float,
    initial_spawn_distance: float | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    n: int = 1,
) -> np.ndarray:
    """Generate a new point in a random direction that is a certain distance from other points.

    Parameters
    ----------
    points : np.ndarray
        NxM array of coordinates of all old points,
        where N is the number of points and M the dimension
    minimum_distance : float
        Minimum distance for the newly generated
        point to be from all other points
    initial_spawn_distance : float | None
        Initial distance from origin.
        If None, is the same as minimum_distance. Default = None
    rtol : float
        Relative tolerance. Default: 1e-5
    atol : float
        Absolute tolerance. Default: 1e-8
    n : int
        number of points to generate. Default: 1

    Returns
    -------
    r_vec : np.ndarray
        Array of length M of coordinates of new point.
        Shape n*M if multiple points are being generated.

    Raises
    ------
    ValueError
        If `minimum_distance` or `initial_spawn_distance` is less or equal to 0

    """
    # If points is only a 1d array (when only one initial point is supplied)
    # make sure it is 2d so we index it correctly.
    points = np.atleast_2d(points)

    if n > 1:
        n_initial_points = np.shape(points)[0]

        new_points = np.zeros((n_initial_points + n, 3), dtype=float)

        # Add original points into this array
        new_points[:n_initial_points, :] = points

        # Overwrite original points
        points = new_points
        for i in range(n):
            new_point = sample_new_point(
                points[: n_initial_points + i, :],
                minimum_distance,
                initial_spawn_distance=initial_spawn_distance,
                rtol=rtol,
                atol=atol,
            )
            points[n_initial_points + i, :] = new_point
        return points[n_initial_points:, :]

    minimum_distance = float(minimum_distance)

    if initial_spawn_distance is None:
        initial_spawn_distance = minimum_distance

    if minimum_distance <= 0.0 or initial_spawn_distance <= 0.0:
        raise ValueError()

    r_unit = get_random_unit_vector()
    r_vec = r_unit * initial_spawn_distance

    distances = np.linalg.norm(points - r_vec, axis=1)
    while not np.isclose(np.min(distances), minimum_distance, rtol=rtol, atol=atol):
        min_distance_position = points[np.argmin(distances), :]
        r_length = np.linalg.norm(r_vec)

        b = 2.0 * r_length - 2.0 * np.dot(min_distance_position, r_unit)
        c = (
            -2.0 * r_length * np.dot(min_distance_position, r_unit)
            + np.linalg.norm(min_distance_position) ** 2
            - minimum_distance**2
            + r_length**2
        )

        roots = np.polynomial.polynomial.polyroots([c, b, 1.0])
        k = (roots[roots >= 0.0]).min()
        r_vec += k * r_unit

        distances = np.linalg.norm(points - r_vec, axis=1)
    return r_vec


def icosahedron_unit_sphere(level: int = 0, subdivision: int = 2) -> np.ndarray:
    """Get vertices of an icosahedron for even sampling of a unit sphere.

    Teanby et al, 2006. https://sci-hub.se/https://doi.org/10.1016/j.cageo.2006.01.007
    Recursive. Might be

    Parameters
    ----------
    level : int
        Level. Default = 1
    subdivision : int
        Prime integer, currently only 2 implemented. Default = 2

    Returns
    -------
    vertices : np.ndarray
        Numpy array of shape Nx3, with N vertices

    Raises
    ------
    NotImplementedError
        If `subdivision` is not 2
    ValueError
        If `level` is less than 0.

    """
    if subdivision != 2:
        msg = f"Only subdivision == 2 (bisection) is implemented at the moment, but was {subdivision}"
        raise NotImplementedError(msg)

    if subdivision not in {2, 3, 5, 7, 11}:
        # Check only up to 11 for now.
        msg = f"Subdivision needs to be a prime integer, but was {subdivision}"
        raise ValueError(msg)

    if level < 0:
        raise ValueError()

    if level == 0:
        phi = 2.0 * np.cos(np.pi / 5.0)
        vertices = np.array(
            [
                [0.0, phi, 1.0],
                [0.0, -phi, 1.0],
                [0.0, phi, -1.0],
                [0.0, -phi, -1.0],
                [1.0, 0.0, phi],
                [-1.0, 0.0, phi],
                [1.0, 0.0, -phi],
                [-1.0, 0.0, -phi],
                [phi, 1.0, 0.0],
                [-phi, 1.0, 0.0],
                [phi, -1.0, 0.0],
                [-phi, -1.0, 0.0],
            ]
        )
        normalization = 1.0 / np.sqrt(1.0 + 4.0 * (np.cos(np.pi / 5.0)) ** 2)
        return vertices * normalization

    vertices_below = icosahedron_unit_sphere(level - 1, subdivision=subdivision)
    hull = ConvexHull(points=vertices_below, incremental=False)
    triangle_indices = hull.simplices

    ntriangles = np.shape(triangle_indices)[0]

    new_points = np.empty(shape=(ntriangles * 3, 3))
    for triangle_idx in range(ntriangles):
        vertices_of_triangle = triangle_indices[triangle_idx, :]

        edges = itertools.combinations(iterable=vertices_of_triangle, r=2)

        new_points[triangle_idx * 3 : triangle_idx * 3 + 3, :] = np.array(
            [
                project_on_unit_sphere(np.sum(vertices_below[edge, :], axis=0))
                for edge in edges
            ]
        )
    vertices = np.unique(np.append(vertices_below, new_points, axis=0), axis=0)
    return vertices


def get_minimum_site_idx_for_binding_sites(
    binding_site_sets: list[set[int]],
    energies: list[float] | np.ndarray,
) -> list[int]:
    """Get the indices with the minimum energy out of sets of binding sites.

    Parameters
    ----------
    binding_site_sets : list[set[int]]
        List of sets of binding sites.
    energies : list[float] | np.ndarray
        Energies of each binding site.

    Returns
    -------
    minimum_for_each_set : list[int]
        Index for each binding site set corresponding to the minimum of that set.

    Raises
    ------
    ValueError
        If the length of `energies` is not the same as the total number of
        indices in `binding_site_sets`.

    """
    total_binding_sites = sum(
        len(binding_site_set) for binding_site_set in binding_site_sets
    )
    if len(energies) != total_binding_sites:
        msg = f"Length of energies ({len(energies)}) was not the same as the length of sites (total_binding_sites)"
        raise ValueError(msg)
    energies = np.asarray(energies)

    minimum_for_each_set = []
    for binding_site_set in binding_site_sets:
        binding_site_list = list(binding_site_set)
        min_index = np.argmin(energies[binding_site_list])
        minimum_for_each_set.append(binding_site_list[min_index])
    return minimum_for_each_set


def unwrap_trajectory_from_displacement(
    positions: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Unwrap trajectory based on displacement between frames.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape Nx3 of the wrapped positions over N frames.
    cell : np.ndarray
        3x3 array or 1d array of diagonal of cell lengths.

    Returns
    -------
    unwrapped_positions : np.ndarray
        Unwrapped positions.

    """
    if cell.ndim == 2:
        cell = np.diag(cell)
    diff = np.diff(positions, axis=0)
    diff = np.insert(diff, 0, 0, axis=0)
    is_currently_in_image = np.cumsum(np.sign(np.round(diff / (cell * 0.5))), axis=0)
    unwrapped_positions = positions - cell * is_currently_in_image
    return unwrapped_positions


def set_up_periodic_plot(
    box_size: np.ndarray, ax: plt.Axes | None = None, additional_width: float = 0.1
) -> plt.Axes:
    """Set up a plot of a periodic box.

    Parameters
    ----------
    box_size : np.ndarray
        Size of the box. 3x3 array or 1d array of the diagonal.
    ax : plt.Axes | None
        Axes to plot on. If None, uses :func:`matplotlib.pyplot.gca()`.
        Default = None.
    additional_width : float
        Additional width (margins) to include around the central image.
        Default = 0.1.

    Returns
    -------
    ax : plt.Axes
        Axes to plot on.

    """
    if ax is None:
        ax = plt.gca()

    ax.set_aspect("equal")
    ax.set_xlabel("x (Angstrom)")
    ax.set_ylabel("y (Angstrom)")

    if box_size.ndim == 2:
        box_size = np.diag(box_size)

    x_extra = box_size[0] * additional_width
    y_extra = box_size[1] * additional_width
    ax.set_xlim((-x_extra, box_size[0] + x_extra))
    ax.set_ylim((-y_extra, box_size[1] + y_extra))

    box = patches.Rectangle(
        (0, 0), box_size[0], box_size[1], edgecolor="k", facecolor="none", lw=0.25
    )
    ax.add_patch(box)

    return ax


def plot_periodic_images(
    x: np.ndarray | list,
    y: np.ndarray | list,
    box_size: np.ndarray,
    ax: plt.Axes | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> plt.Axes:
    """Plot periodic images of the data as well.

    Parameters
    ----------
    x : np.ndarray | list
        X-data
    y : np.ndarray | list
        Y-data
    box_size : np.ndarray
        Size of the box. 3x3 array or 1d array of the diagonal.
    ax : plt.Axes | None
        Axes to plot on. If None, uses :func:`matplotlib.pyplot.gca()`.
        Default = None.
    plot_kwargs : dict[str, Any] | None
        Keyword arguments passed to :meth:`matplotlib.pyplot.axes.Axes.plot`.
        Default = None.

    Returns
    -------
    ax : plt.Axes
        Axes that was plotted on.

    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if box_size.ndim == 2:
        box_size = np.diag(box_size)
    if ax is None:
        ax = plt.gca()
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(y, list):
        y = np.asarray(y)

    if np.any(np.abs(x) > box_size[0]) or np.any(np.abs(y) > box_size[1]):
        # Outside of first periodic image
        images_to_include: tuple[int, ...] = (-2, -1, 0, 1, 2)
    else:
        images_to_include = (-1, 0, 1)

    images = itertools.product(images_to_include, images_to_include)
    for image in images:
        image_x = x + image[0] * box_size[0]
        image_y = y + image[1] * box_size[1]
        ax.plot(image_x, image_y, **plot_kwargs)

    return ax


def get_tip4p_water() -> Atoms:
    x = angleHOH * np.pi / 180 / 2
    pos = [
        [0, 0, 0],
        [0, rOH * np.cos(x), rOH * np.sin(x)],
        [0, rOH * np.cos(x), -rOH * np.sin(x)],
    ]
    water = Atoms("OH2", positions=pos)
    return water
