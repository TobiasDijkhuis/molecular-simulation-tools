"""Collection of geometry tools."""

import sys
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum, auto, unique
from itertools import product
from typing import TypeAlias

import numpy as np
from ase import Atoms
from ase.constraints.fix_atoms import FixAtoms
from ase.geometry import find_mic, get_angles
from ase.neighborlist import NeighborList, natural_cutoffs

from molecular_simulation_tools.connectivity import (
    identify_molecules,
)
from molecular_simulation_tools.utils import (
    check_same_number_of_atoms,
    combine_overlapping_sets,
    correct_distance_for_pbc,
    get_permutations_exchange_identical_atoms,
    unwrap_trajectory_from_displacement,
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
        atoms.positions - center, cell=atoms.cell, pbc=atoms.pbc
    )
    indices_within_radius = np.flatnonzero(vlen <= radius)
    return indices_within_radius, relative_positions


def get_periodic_images_within_radius(
    atoms: Atoms,
    center: np.ndarray,
    radius: float,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Return atom indices and periodic translation vectors inside a sphere.

    Each returned pair ``(index, translation)`` identifies a particular
    periodic image of an atom.
    """
    if not atoms.pbc.any():
        distances = atoms.positions - center
        mask = np.linalg.norm(distances, axis=1) <= radius
        indices = np.nonzero(mask)[0].tolist()
        translations = [(0, 0, 0) for _ in indices]
        return indices, translations

    cell = atoms.cell.array
    pbc = np.asarray(atoms.pbc)

    # Conservative number of cells to search in each direction.
    # This is deliberately somewhat generous.
    lengths = np.linalg.norm(cell, axis=1)
    n_images = np.ceil(radius / lengths).astype(int) + 1

    ranges = [
        range(-n_images[i], n_images[i] + 1) if pbc[i] else range(0, 1)
        for i in range(3)
    ]

    selected_indices = []
    selected_translations = []

    for image in product(*ranges):
        image = np.asarray(image, dtype=int)

        positions = atoms.positions + image @ cell
        distances = positions - center

        mask = np.linalg.norm(distances, axis=1) <= radius

        for index in np.nonzero(mask)[0]:
            selected_indices.append(int(index))
            selected_translations.append(tuple(image.copy()))

    return selected_indices, selected_translations


def make_cutout(
    atoms: Atoms,
    selected: Iterable[tuple[int, tuple[int, int, int]]],
) -> Atoms:
    positions = []
    numbers = []

    cell = atoms.cell.array

    for index, image in selected:
        translation = np.asarray(image) @ cell

        positions.append(atoms.positions[index] + translation)
        numbers.append(atoms.numbers[index])

    cutout = Atoms(
        numbers=numbers,
        positions=positions,
        pbc=False,
    )

    return cutout


def add_molecule_image(
    selected: set[tuple[int, tuple[int, int, int]]],
    molecule: list[int],
    image: tuple[int, int, int],
) -> None:
    for index in molecule:
        selected.add((index, image))


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
    # if (
    #     atoms.cell is not None
    #     and not np.all(atoms.cell == 0)
    #     and radius >= 0.5 * np.min(atoms.cell.lengths())
    # ):
    #     msg = f"Radius of {radius} Angstrom is too big for cell. Would include same atoms multiple times."
    #     raise ValueError(msg)

    indices_within_radius, images = get_periodic_images_within_radius(
        atoms, center, radius
    )

    if len(indices_within_radius) == 0:
        msg = f"No atoms were found within radius {radius} Angstrom of center {center}"
        raise RuntimeError(msg)

    molecules = identify_molecules(atoms, cutoffs=cutoffs)
    molecule_for_atom = get_molecule_idx_of_atom(molecules, range(len(atoms)))

    selected: set[tuple[int, tuple[int, int, int]]] = set()
    for index, translation in zip(indices_within_radius, images, strict=True):
        # Find molecule containing `index`
        molecule = list(molecules[molecule_for_atom[index]])

        if keep_molecules_intact:
            add_molecule_image(selected, molecule, translation)
        else:
            selected.add(
                (index, (int(translation[0]), int(translation[1]), int(translation[2])))
            )
    # if keep_molecules_intact:
    #     indices_within_radius = complete_intact_molecules(
    #         atoms,
    #         indices_within_radius,
    #         allowed_molecules=allowed_molecules,
    #         cutoffs=cutoffs,
    #     )

    cutout_atoms = make_cutout(atoms, selected)

    return cutout_atoms


def get_molecule_idx_of_atom(
    molecules: list[np.ndarray], atom_indices: Iterable[int]
) -> dict[int, int]:
    molecule_for_atom: dict[int, int] = {}
    for atom_idx in atom_indices:
        in_molecule_idx = [
            i for i, molecule in enumerate(molecules) if atom_idx in molecule
        ][0]
        molecule_for_atom[atom_idx] = in_molecule_idx
    return molecule_for_atom


def cut_out_trajectory_within_radius(
    frames: list[Atoms],
    centers: list[np.ndarray] | np.ndarray | int,
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
    centers : list[np.ndarray] | np.ndarray | int
        Center of sphere to be cut. If a list of arrays, the center changes between
        frames. If an integer, the index of the atom that is the center of the sphere.
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
    elif isinstance(centers, int):
        centers = [frame.positions[centers, :] for frame in frames]
        print(centers)
        centers = unwrap_trajectory_from_displacement(
            np.asarray(centers), frames[0].cell
        )
        print(centers)

    selected: set[tuple[int, tuple[int, int, int]]] = set()
    for frame, center in zip(frames, centers, strict=True):
        indices, images = get_periodic_images_within_radius(frame, center, radius)

        molecules = identify_molecules(frame, cutoffs=cutoffs)
        molecule_for_atom = get_molecule_idx_of_atom(molecules, range(len(frame)))

        for index, translation in zip(indices, images, strict=True):
            molecule = list(molecules[molecule_for_atom[index]])

            if keep_molecules_intact:
                add_molecule_image(selected, molecule, translation)
            else:
                selected.add(
                    (
                        index,
                        (int(translation[0]), int(translation[1]), int(translation[2])),
                    )
                )

    cutout_atoms = [make_cutout(frame, selected) for frame in frames]
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


def get_two_dimensional_distances(
    x: float,
    y: float,
    point_coordinates: np.ndarray,
    box_size: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate the two-dimensional projected distance, including PBCs.

    Parameters
    ----------
    x : float
        x-coordinate of desired point
    y : float
        y-coordinate of desired point
    point_coordinates : np.ndarray
        Nx2 (or more) array of N point coordinates.
    box_size : np.ndarray | None
        Size of the box. If None, do not include periodic boundary conditions.
        Default = None.

    Returns
    -------
    distances : np.ndarray
        Array with two-dimensional distances from ``(x, y)`` to the N points in
        `point_coordinates`.

    """
    point_coordinates = np.atleast_2d(point_coordinates)
    delta_x = point_coordinates[:, 0] - x
    delta_y = point_coordinates[:, 1] - y

    if box_size is not None:
        delta_x = correct_distance_for_pbc(delta_x, box_size[0])
        delta_y = correct_distance_for_pbc(delta_y, box_size[1])

    distances = np.sqrt(delta_x**2 + delta_y**2)
    return distances


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
    two_dimensional_distances = get_two_dimensional_distances(
        x, y, point_coordinates, box_size=box_size
    )

    in_cylinder = (distance - two_dimensional_distances) >= 0.0

    if not np.any(in_cylinder):
        msg = f"No points found within a radius of {distance} of ({x}, {y}) in a 2D projection"
        raise ValueError(msg)

    necessary_delta_z = np.sqrt(
        distance**2 - two_dimensional_distances[in_cylinder] ** 2
    )

    height = np.max(point_coordinates[in_cylinder, 2] + necessary_delta_z)

    return height


def find_min_height_for_adsorbate_on_surface(
    surface: Atoms,
    adsorbate_origin: np.ndarray,
    distance: float,
    adsorbate: Atoms | None = None,
) -> float:
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
    adsorbate_origin : np.ndarray
        Origin of adsorbate
    distance : float
        Distance from all other atoms.
    adsorbate : Atoms | None
        Adsorbate. If None, create a dummy Atoms instance with one atom.
        Default = None.

    Returns
    -------
    max_height : float
        Height to place adsorbate at for it to be `distance` from all atoms
        in the surface.

    """
    if adsorbate is None:
        # Dummy, single atom
        adsorbate = Atoms("H", positions=[[0, 0, 0]])
    else:
        adsorbate = adsorbate.copy()
        adsorbate.set_center_of_mass([0, 0, 0])

    cell = surface.cell if any(surface.pbc) else None
    max_height = sys.float_info.min
    for atom_idx in range(len(adsorbate)):
        pos = adsorbate.positions[atom_idx, :]

        height = find_min_height_for_distance(
            pos[0] + adsorbate_origin[0],
            pos[1] + adsorbate_origin[1],
            surface.positions,
            distance,
            box_size=cell,
        )
        max_height = max(max_height, height)
    return max_height


def find_min_height_for_adsorbate_on_surface_grid(
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
    grid_x, grid_y = construct_grid_in_cell(surface.get_cell(), ngrid)
    if isinstance(ngrid, int):
        ngrid = (ngrid, ngrid)

    sample_heights = np.empty((ngrid[0], ngrid[1]))
    for i_x in range(ngrid[0]):
        for i_y in range(ngrid[1]):
            max_height = find_min_height_for_adsorbate_on_surface(
                surface,
                np.asarray([grid_x[i_x, i_y], grid_y[i_x, i_y]]),
                distance,
                adsorbate=adsorbate,
            )
            sample_heights[i_x, i_y] = max_height
    return grid_x, grid_y, sample_heights


def calculate_rmsd(
    atoms: Atoms,
    target: Atoms,
    indices: int | list[int] | None,
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
    indices : int | list[int] | None
        Single index, list of indices, or None if all atoms.
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
    if isinstance(indices, int):
        indices = [indices]
    n_atoms_for_rmsd = len(indices)

    original_positions = atoms.get_positions()
    target_positions = target.get_positions()

    if permute:
        permutations = get_permutations_exchange_identical_atoms(atoms, indices=indices)
    else:
        permutations = [indices.copy()]

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


def course_grain_binding_sites(
    sites: list[Atoms],
    indices: int | list[int] | None,
    max_rmsd: int | float = 10,
    permute: bool = True,
    top_down: bool = False,
) -> list[set[int]]:
    """Course grain binding sites that are near to eachother.

    Parameters
    ----------
    sites : list[Atoms]
        List of geometries of surfaces with adsorbates.
    indices : int | list[int] | None
        Indices to be used to calculate the RMSD. If None, use all.
    max_rmsd : int | float
        Maximum RMSD to be considered the same site. Default = 0.5 Angstrom.
    permute : bool
        Whether to permute indices in `indices` with identical atoms. Default = True.
    top_down : bool
        Only take into account the two-dimensional projected distance.
        If True, `indices` needs to be passed as a single integer.
        Default = False.

    Returns
    -------
    binding_site_sets : list[set[int]]
        List of binding sites that are considered the same.

    Raises
    ------
    TypeError
        If `top_down` is True, but `indices` is not given, or is not a single
        integer.

    """
    if top_down and indices is None or not isinstance(indices, int):
        raise TypeError

    binding_site_sets: list[set[int]] = []
    for frame_idx, frame in enumerate(sites):
        for other_frame_idx in range(frame_idx + 1, len(sites)):
            other_frame = sites[other_frame_idx]

            if top_down:
                rmsd = get_two_dimensional_distances(
                    frame.positions[indices, 0],
                    frame.positions[indices, 1],
                    other_frame.positions[indices, :],
                    box_size=np.diag(frame.cell),
                )[0]
            else:
                rmsd, _ = calculate_rmsd(frame, other_frame, indices, permute=permute)
            if rmsd > max_rmsd:
                continue

            new_set = True
            for binding_site_set in binding_site_sets:
                if frame_idx in binding_site_set or other_frame_idx in binding_site_set:
                    binding_site_set.add(frame_idx)
                    binding_site_set.add(other_frame_idx)
                    new_set = False
            if new_set:
                binding_site_sets.append({frame_idx, other_frame_idx})

        saw_this_frame = False
        for binding_site_set in binding_site_sets:
            if frame_idx in binding_site_set:
                saw_this_frame = True
                break
        if not saw_this_frame:
            binding_site_sets.append({frame_idx})

    binding_site_sets = combine_overlapping_sets(binding_site_sets)
    return binding_site_sets


@unique
class HydrogenBondingType(StrEnum):
    NONE = auto()
    A = auto()
    AA = auto()
    AAA = auto()
    D = auto()
    DA = auto()
    DAA = auto()
    DAAA = auto()
    DD = auto()
    DDA = auto()
    DDAA = auto()
    DDAAA = auto()
    DDD = auto()
    DDDA = auto()
    DDDAA = auto()


WaterHydrogenBondingNetwork: TypeAlias = list[tuple[int, int, int]]


def get_hydrogen_bonding_network(
    atoms: Atoms,
    max_hydrogen_bond_length: float = 3.0,
    max_hydrogen_bond_angle: float = 20.0,
) -> WaterHydrogenBondingNetwork:
    cutoffs_hydrogen_bonds = natural_cutoffs(
        atoms, H=max_hydrogen_bond_length / 2.0, O=max_hydrogen_bond_length / 2.0
    )
    nl_hydrogen_bonds = NeighborList(
        cutoffs_hydrogen_bonds, self_interaction=False, bothways=True, skin=0.0
    )
    nl_hydrogen_bonds.update(atoms)
    cm_hydrogen = nl_hydrogen_bonds.get_connectivity_matrix(sparse=False)

    cutoffs_covalent_bonds = natural_cutoffs(atoms, O=0.9)
    nl_covalent_bonds = NeighborList(
        cutoffs_covalent_bonds, self_interaction=False, bothways=True, skin=0.0
    )
    nl_covalent_bonds.update(atoms)
    cm_covalent = nl_covalent_bonds.get_connectivity_matrix(sparse=False)

    cm_non_covalent_hydrogen = cm_hydrogen & ~cm_covalent

    symbols = atoms.symbols
    positions = atoms.positions
    network: WaterHydrogenBondingNetwork = []
    for hydrogen_idx in np.flatnonzero(symbols == "H"):
        covalent_oxygen_indices = np.flatnonzero(
            cm_covalent[:, hydrogen_idx] & (symbols == "O")
        )

        if len(covalent_oxygen_indices) != 1:
            msg = f"Expected hydrogen to be bound to one oxygen, but got {len(covalent_oxygen_indices)} oxygens"
            raise ValueError(msg)

        donor_idx = int(covalent_oxygen_indices[0])

        acceptor_indices = np.flatnonzero(
            cm_non_covalent_hydrogen[:, hydrogen_idx] & (symbols == "O")
        )
        for acceptor_idx in acceptor_indices:
            angle = get_angles(
                np.atleast_2d(positions[donor_idx, :] - positions[hydrogen_idx, :]),
                np.atleast_2d(positions[acceptor_idx, :] - positions[hydrogen_idx, :]),
                cell=atoms.cell,
                pbc=atoms.pbc,
            )[0]
            if 180 - angle <= max_hydrogen_bond_angle:
                network.append((donor_idx, int(hydrogen_idx), int(acceptor_idx)))

    return network


def get_hydrogen_bonding_type(
    atomic_indices_in_molecule: Iterable[int], network: WaterHydrogenBondingNetwork
) -> HydrogenBondingType:
    donors = Counter(hbond[0] for hbond in network)
    acceptors = Counter(hbond[2] for hbond in network)

    donor_count = sum(donors[atom_idx] for atom_idx in atomic_indices_in_molecule)
    acceptor_count = sum(acceptors[atom_idx] for atom_idx in atomic_indices_in_molecule)

    if donor_count == 0 and acceptor_count == 0:
        return HydrogenBondingType.NONE
    type_string = "d" * donor_count + "a" * acceptor_count
    return HydrogenBondingType(type_string)


def get_all_hydrogen_bonding_counts(
    molecules: Iterable[Iterable[int]], network: WaterHydrogenBondingNetwork
) -> Counter[HydrogenBondingType]:
    return Counter(
        get_hydrogen_bonding_type(molecule, network) for molecule in molecules
    )
