"""Collection of geometry tools."""

import itertools
import sys

import numpy as np
from ase import Atoms
from ase.geometry import find_mic


def construct_grid_in_cell(
    cell: np.ndarray, ngrid: int | tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Create a grid of points sampled equally in the cell.

    Parameters
    ----------
    cell : np.ndarray
        3x3 matrix of cell dimensions.
    ngrid : int | tuple[int, int]
        Number of grid points to sample along x and y. If an integer,
        takes the same number of points along x and y.

    Returns
    -------
    X : np.ndarray
        Grid of x coordinates
    Y : np.ndarray
        Grid of y coordinates

    Raises
    ------
    ValueError
        If `ngrid` is not a 2-tuple.

    """
    if isinstance(ngrid, int):
        ngrid = (ngrid, ngrid)

    if len(ngrid) != 2:
        msg = f"Number of dimensions of ngrid needs to be 2, but was {len(ngrid)}"
        raise ValueError(msg)

    x = discretize_cell_length(cell[0, 0], ngrid[0])
    y = discretize_cell_length(cell[1, 1], ngrid[1])
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    return X, Y


def discretize_cell_length(length: int | float, ngrid: int) -> np.ndarray:
    """Discretize the length of a cell.

    Places the points such that they are all equidistant, including the first and
    last point, if you include periodic boundary conditions.

    Parameters
    ----------
    length : int | float
        Length to discretize
    ngrid : int
        Number of points

    Returns
    -------
    np.ndarray
        array containing equidistant points along the length `length`.

    """
    spacing = float(length) / ngrid
    return np.linspace(spacing / 2, length - spacing / 2, num=ngrid, endpoint=True)


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


def find_min_height_for_distance(
    x: float,
    y: float,
    point_coordinates: np.ndarray,
    distance: float,
    box_size: np.ndarray | None = None,
) -> float:
    """Find the minimum height for a point to be `distance` away from other points.

    Calculate the minimum height ``z`` for a point ``(x, y)`` for it to be at least
    `distance` away from all points in `point_coordinates`.

    Parameters
    ----------
    x : float
        x-coordinate of desired point
    y : float
        y-coordinate of desired point
    point_coordinates : np.ndarray
        Nx2 (or more) array of N point coordinates.
    distance : float
        Minimum distance to all other points.
    box_size : np.ndarray | None
        Size of the box. If None, do not include periodic boundary conditions.
        Default = None.

    Returns
    -------
    height : float
        Minimum height required for ``(x, y)`` to be `distance` away from all other
        points.

    Raises
    ------
    ValueError
        If no points within radius `distance` from ``(x, y)``
        are found in `point_coordinates` (in a 2D projection).

    """
    delta_x = point_coordinates[:, 0] - x
    delta_y = point_coordinates[:, 1] - y

    if box_size is not None:
        delta_x = correct_distance_for_pbc(delta_x, box_size[0])
        delta_y = correct_distance_for_pbc(delta_y, box_size[1])

    delta_x_squared = delta_x**2
    delta_y_squared = delta_y**2

    in_cylinder = (distance**2 - delta_x_squared - delta_y_squared) >= 0.0

    if not np.any(in_cylinder):
        msg = f"No points found within a radius of {distance} of ({x}, {y}) in a 2D projection"
        raise ValueError(msg)

    point_coordinates = point_coordinates[in_cylinder, :]
    delta_x_squared = delta_x_squared[in_cylinder]
    delta_y_squared = delta_y_squared[in_cylinder]

    necessary_delta_z_squared = distance**2 - delta_x_squared - delta_y_squared

    height = np.max(point_coordinates[:, 2] + np.sqrt(necessary_delta_z_squared))

    return height


def find_min_height_for_adsorbate_on_surface(
    surface: Atoms,
    ngrid: int | tuple[int, int],
    distance: float,
    adsorbate: Atoms | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the minimum height to place an adsorbate at least `distance` away from surface atoms.

    Calculate the minimum height for `adsorbate` for it to have all its
    atoms at least `distance` away from the other atoms in `surface`.

    The returned points correspond to placing the center-of-mass of `adsorbate`
    at ``(x, y, z)``, such that the minimum distance to any atom in the surface is
    `distance`.

    Parameters
    ----------
    surface : Atoms
        Surface
    ngrid : int | tuple[int, int]
        Number of grid points along x and y
    distance : float
        Distance from all other atoms.
    adsorbate : Atoms | None
        Adsorbate. If None, create a dummy Atoms instance with one atom.
        Default = None.

    Returns
    -------
    grid_x : np.ndarray
        grid of positions along x
    grid_y : np.ndarray
        grid of positions along y
    sample_heights : np.ndarray
        height of the points corresponding to the points in `grid_x` and `grid_y.

    """
    if adsorbate is None:
        # Dummy, single atom
        adsorbate = Atoms("H", [[0, 0, 0]])
    else:
        adsorbate = adsorbate.copy()
        adsorbate.set_center_of_mass([0, 0, 0])

    n_atoms_in_adsorbate = len(adsorbate)

    grid_x, grid_y = construct_grid_in_cell(surface.get_cell(), ngrid)
    if isinstance(ngrid, int):
        ngrid = (ngrid, ngrid)

    sample_heights = np.empty((ngrid[0], ngrid[1]))
    cell = np.diag(surface.get_cell())
    for i_x in range(ngrid[0]):
        for i_y in range(ngrid[1]):
            max_height = sys.float_info.min

            for atom_idx in range(n_atoms_in_adsorbate):
                pos = adsorbate.positions[atom_idx, :]

                height = find_min_height_for_distance(
                    pos[0] + grid_x[i_x, i_y],
                    pos[1] + grid_y[i_x, i_y],
                    surface.positions,
                    distance,
                    box_size=cell,
                )
                max_height = max(max_height, height)
            sample_heights[i_x, i_y] = max_height
    return grid_x, grid_y, sample_heights


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


def calculate_rmsd(
    atoms: Atoms,
    target: Atoms,
    indices: list[int] | None,
    permute: bool = True,
    return_permuted_target: bool = True,
) -> tuple[float, tuple[Atoms, Atoms]]:
    """Calculate the root-mean-squared-displacement (rmsd) between atoms `atoms` and `target`.

    Parameters
    ----------
    atoms : Atoms
        atoms
    target : Atoms
        target
    indices : list[int] | None
        List of indices, or None if all atoms.
    permute : bool
        Whether to permute the atom indices to get the minimum
        rmsd. Only permutes between identical elements. Default = True.
    return_permuted_target : bool
        Whether to return the permuted target,
        that has the minimum rmsd. Default = True.

    Returns
    -------
    min_rmsd : float
        minimum rmsd
    return_atoms : tuple[Atoms, Atoms]
        tuple of `atoms` and `target` with only the indices in `indices`.
        If `return_permuted_target` is True, returns the permuted `target` that leads
        to the minimum rmsd.

    Raises
    ------
    ValueError
        If the number of atoms, order of elements or cells are different between
        `atoms` and `target`.

    """
    if len(atoms) != len(target):
        msg = f"Number of atoms in atoms ({len(atoms)}) was not the same as the number of atoms in target ({len(target)})"
        raise ValueError(msg)
    if np.any(atoms.cell != target.cell):
        msg = f"Cell of atoms ({atoms.cell}) is not the same as the cell of target ({target.cell})"
        raise ValueError(msg)
    if np.any(atoms.symbols != target.symbols):
        msg = f"atoms does not have the same symbols ({atoms.symbols}) as target ({target.symbols})"
        raise ValueError(msg)

    if indices is None:
        indices = list(range(len(atoms)))
    n_atoms_for_rmsd = len(indices)

    original_positions = atoms.get_positions()
    target_positions = target.get_positions()

    if permute:
        permutations: itertools.permutations | tuple[list[int]] = (
            itertools.permutations(indices)
        )
    else:
        permutations = (indices.copy(),)

    min_rmsd = sys.float_info.max
    for permutation in permutations:
        permutation = list(permutation)
        if not np.all(atoms.numbers[indices] == target.numbers[permutation]):
            # Only exchange identical atoms
            continue

        _, distances = find_mic(
            original_positions[indices, :] - target_positions[permutation, :],
            atoms.cell,
            atoms.pbc,
        )

        rmsd = np.sqrt(np.sum(distances**2) / n_atoms_for_rmsd)

        min_rmsd = min(min_rmsd, rmsd)
    if return_permuted_target:
        return_atoms = (atoms[indices], target[permutation])
    else:
        return_atoms = (atoms[indices], target[indices])
    return min_rmsd, return_atoms
