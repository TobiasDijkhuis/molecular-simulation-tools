"""Collection of geometry tools."""

import itertools
import sys

import numpy as np
from ase import Atoms
from ase.constraints.fix_atoms import FixAtoms
from ase.geometry import find_mic
from scipy.spatial import ConvexHull

from molecular_simulation_tools.connectivity import (
    complete_intact_molecules,
)
from molecular_simulation_tools.utils import (
    check_same_number_of_atoms,
    correct_distance_for_pbc,
    get_random_unit_vector,
    project_on_unit_sphere,
    get_permutations_exchange_identical_atoms,
)


def get_constraint_outside_radius(
    atoms: Atoms, center: np.ndarray, radius: float
) -> FixAtoms:
    """Get a constraint that fixes atoms outside a certain radius.

    Parameters
    ----------
    atoms : Atoms
        Atoms to keep (un)constrained
    center : np.ndarray
        Center of sphere
    radius : float
        Radius to keep unconstrained

    Returns
    -------
    FixAtoms
        Constraint that fixes atoms outside radius `radius` from center `center`.

    """
    _, vlen = find_mic(atoms.positions - center, cell=atoms.cell, pbc=True)
    return FixAtoms(mask=vlen > radius)


def get_atoms_indices_within_radius(
    atoms: Atoms, center: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get the indices of all atoms within `radius` of `center`.

    Parameters
    ----------
    atoms : Atoms
        Atoms to analyze
    center : np.ndarray
        Center of sphere
    radius : float
        Radius

    Returns
    -------
    indices_within_radius : np.ndarray
        Array of indices within `radius` of `center`.
    relative_positions : np.ndarray
        Relative positions of all atoms wrt `center`

    """
    relative_positions, vlen = find_mic(
        atoms.positions - center, cell=atoms.cell, pbc=True
    )
    indices_within_radius = np.flatnonzero(vlen <= radius)
    return indices_within_radius, relative_positions


def cut_out_atoms_within_radius(
    atoms: Atoms,
    center: np.ndarray,
    radius: float,
    keep_molecules_intact: bool = True,
    allowed_molecules: list[str] | set[str] | None = None,
    cutoffs: dict[str, float] | None = None,
) -> Atoms:
    """Cut out atoms within a radius from some position.

    Parameters
    ----------
    atoms : Atoms
        Atoms to be cut out
    center : np.ndarray
        Center of sphere to be cut
    radius : float
        Radius of sphere to be cut in Angstrom.
    keep_molecules_intact : bool
        Identify molecules using :func:`identify_molecules`, and keep each molecule
        intact (i.e. if any of the atoms are inside the sphere, keep the whole molecule).
        Requires networkx to be installed. Default = True.
    allowed_molecules : list[str] | set[str] | None
        List of elementary compositions of allowed molecules. If not None, check
        that only allowed molecules are present using :func:`check_only_allowed_molecules`.
        Default = None.
    cutoffs : dict[str, float] | None
        Cutoffs for each element. Dictionary with keys for the symbols and values
        of the cutoff radii. If None, use the :data:`ase.data.covalent_radii`. Default: None

    Returns
    -------
    cutout_atoms : Atoms
        Cut out atoms, with their ``pbc`` and ``cell`` attributes set to False and None.

    Raises
    ------
    ValueError
        If `radius` is more than half the minimum cell length of `atoms`.
    RuntimeError
        If a no atoms are found within `radius` of `center`.
    RuntimeError
        If a non-allowed molecule is within `radius` of `center`.

    """
    if (
        atoms.cell is not None
        and not np.all(atoms.cell == 0)
        and radius >= 0.5 * np.min(atoms.cell.lengths())
    ):
        msg = f"Radius of {radius} Angstrom is too big for cell. Would include same atoms multiple times."
        raise ValueError(msg)

    indices_within_radius, relative_positions = get_atoms_indices_within_radius(
        atoms, center, radius
    )

    if indices_within_radius.size == 0:
        msg = f"No atoms were found within radius {radius} Angstrom of center {center}"
        raise RuntimeError(msg)

    if keep_molecules_intact:
        indices_within_radius = complete_intact_molecules(
            atoms,
            indices_within_radius,
            allowed_molecules=allowed_molecules,
            cutoffs=cutoffs,
        )

    cutout_atoms = atoms[indices_within_radius]
    cutout_atoms.positions = relative_positions[indices_within_radius, :]

    cutout_atoms.pbc = False
    cutout_atoms.cell = None

    return cutout_atoms


def cut_out_trajectory_within_radius(
    frames: list[Atoms],
    centers: list[np.ndarray] | np.ndarray,
    radius: float,
    keep_molecules_intact: bool = True,
    allowed_molecules: list[str] | set[str] | None = None,
    cutoffs: dict[str, float] | None = None,
) -> list[Atoms]:
    """Cut out atoms from a trajectory, keeping the same atoms in all frames.

    Parameters
    ----------
    frames : list[Atoms]
        Trajectory to cut out.
    centers : list[np.ndarray] | np.ndarray
        Center of sphere to be cut. If a list of arrays, the center changes between
        frames.
    radius : float
        Radius of sphere to be cut in Angstrom.
    keep_molecules_intact : bool
        Identify molecules using :func:`identify_molecules`, and keep each molecule
        intact (i.e. if any of the atoms are inside the sphere, keep the whole molecule).
        Requires networkx to be installed. Default = True.
    allowed_molecules : list[str] | set[str] | None
        List of elementary compositions of allowed molecules. If not None, check
        that only allowed molecules are present using :func:`check_only_allowed_molecules`.
        Default = None.
    cutoffs : dict[str, float] | None
        Cutoffs for each element. Dictionary with keys for the symbols and values
        of the cutoff radii. If None, use the :data:`ase.data.covalent_radii`. Default: None

    Returns
    -------
    cutout_atoms : list[Atoms]
        Cut out atoms, with their ``pbc`` and ``cell`` attributes set to False and None.

    Raises
    ------
    ValueError
        If the shape of `centers` is not correct.

    """
    check_same_number_of_atoms(frames)

    if isinstance(centers, np.ndarray):
        if centers.ndim == 1:
            # Single array, i.e. `center` is fixed.
            centers = [centers] * len(frames)
        elif centers.ndim == 2:
            # Center varies per frame
            if centers.shape[0] != len(frames):
                raise ValueError
            centers = list(centers)

    all_indices: set[int] = set()
    for frame, center in zip(frames, centers, strict=True):
        indices, _relative_positions = get_atoms_indices_within_radius(
            frame, center, radius
        )

        if keep_molecules_intact:
            indices = complete_intact_molecules(
                frame, indices, allowed_molecules=allowed_molecules, cutoffs=cutoffs
            )
        all_indices.update(indices)
    all_indices: list[int] = sorted(all_indices)  # type: ignore[no-redef]

    cutout_atoms = [frame[all_indices] for frame in frames]
    for frame, center in zip(cutout_atoms, centers, strict=True):
        _, relative_positions = get_atoms_indices_within_radius(frame, center, radius)
        frame.positions = relative_positions
        frame.pbc = False
        frame.cell = None

    check_same_number_of_atoms(cutout_atoms)

    return cutout_atoms


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

    """
    if isinstance(ngrid, int):
        ngrid = (ngrid, ngrid)

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
    if np.any(atoms.numbers != target.numbers):
        msg = f"atoms does not have the same symbols ({atoms.symbols}) as target ({target.symbols})"
        raise ValueError(msg)

    if indices is None:
        indices = list(range(len(atoms)))
    n_atoms_for_rmsd = len(indices)

    original_positions = atoms.get_positions()
    target_positions = target.get_positions()

    if permute:
        permutations = get_permutations_exchange_identical_atoms(atoms, indices=indices)
    else:
        permutations = (indices.copy(),)

    min_rmsd = sys.float_info.max
    for permutation in permutations:
        _, distances = find_mic(
            original_positions[indices, :] - target_positions[permutation, :],
            atoms.cell,
            atoms.pbc,
        )

        rmsd = np.sqrt(np.sum(distances**2) / n_atoms_for_rmsd)

        if rmsd < min_rmsd:
            min_rmsd = rmsd
            min_permutation = permutation
    if return_permuted_target:
        return_atoms = (atoms[indices], target[min_permutation])
    else:
        return_atoms = (atoms[indices], target[indices])
    return min_rmsd, return_atoms


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
