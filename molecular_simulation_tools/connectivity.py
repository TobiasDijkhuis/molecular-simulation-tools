"""Tools to identify molecules."""

from collections.abc import Iterable

import numpy as np
from ase import Atoms
from ase.neighborlist import build_neighbor_list, natural_cutoffs

try:
    import networkx as nx

    _nx_available = True
except ImportError:
    _nx_available = False


def _atoms_to_graph(
    atoms: Atoms, cutoffs: dict[str, float] | None = None
) -> "nx.Graph":
    """Convert ASE Atoms into a Graph based on their bond connectivity.

    Requires networkx to be installed. Taken from IPSuite.

    Parameters
    ----------
    atoms : Atoms
        Atoms instance to convert
    cutoffs : dict[str, float] | None
        cutoffs of each atom. Dictionary with keys for the symbols and values of the
        cutoff radii. If None, use the :data:`ase.data.covalent_radii`. Default: None

    Returns
    -------
    graph : nx.Graph
        Connectivity graph

    """
    if cutoffs is not None:
        cutoffs = natural_cutoffs(atoms, **cutoffs)
    # This can be optimized by reusing the NL!
    nl = build_neighbor_list(atoms, self_interaction=False, cutoffs=cutoffs)
    cm = nl.get_connectivity_matrix(sparse=False)
    graph = nx.from_numpy_array(cm)
    for i, atom in enumerate(atoms):
        graph.nodes[i]["element"] = atom.symbol
    return graph


def identify_molecules(
    atoms: Atoms, cutoffs: dict[str, float] | None = None
) -> list[np.ndarray]:
    """Identify molecules in a structure based on the connected subgraphs.

    Requires networkx to be installed. Taken from IPSuite.

    Parameters
    ----------
    atoms : Atoms
        Atoms instance to identify molecules in
    cutoffs : dict[str, float] | None
        cutoffs of each element. Dictionary with keys for the symbols and values
        of the cutoff radii. If None, use the :data:`ase.data.covalent_radii`. Default: None

    Returns
    -------
    c_list : list[np.ndarray]
        Array of lists of connected atom indices

    """
    graph = _atoms_to_graph(atoms, cutoffs=cutoffs)
    components = nx.connected_components(graph)
    c_list = [np.array(list(c)) for c in components]
    return c_list


def check_only_allowed_molecules(
    atoms: Atoms, molecules: list[np.ndarray], allowed_molecules: list[str] | set[str]
) -> list[list[int]] | None:
    """Check that the indices of molecules in `molecules` are only allowed molecules.

    Parameters
    ----------
    atoms : Atoms
        Atoms to check
    molecules : list[np.ndarray]
        List of arrays of indices corresponding to different molecules
    allowed_molecules : list[str] | set[str]
        List of elementary compositions of allowed molecules.

    Returns
    -------
    incorrect_atoms : list[list[int]] | None
        Indices of atoms that are incorrect, or None if none were found.

    """
    incorrect_atoms: list[list[int]] = []
    for molecule in molecules:
        formula = atoms.symbols[molecule].get_chemical_formula(mode="all")
        formula = "".join(sorted(formula))
        if formula not in allowed_molecules:
            print(
                f"Not allowed molecule with symbols '{atoms.symbols[molecule]}' and formula '{formula}' detected."
            )
            incorrect_atoms.append(list(molecule))
            continue
    if not incorrect_atoms:
        return None
    return incorrect_atoms


def complete_intact_molecules(
    atoms: Atoms,
    indices: list[int] | np.ndarray,
    allowed_molecules: list[str] | set[str] | None = None,
    cutoffs: dict[str, float] | None = None,
) -> np.ndarray:
    """Get the indices of atoms to keep `indices` fully connected.

    Create a neighborlist of the original atoms using :func:`identify_molecules`,
    and then make sure that any index in `indices` is kept fully connected
    to its neighbors.

    Parameters
    ----------
    atoms : Atoms
        Atoms to keep some parts fully connected in
    indices : list[int] | np.ndarray
        Indices of atoms to keep connected.
    allowed_molecules : list[str] | set[str] | None
        List of elementary compositions of allowed molecules. If not None, check
        that only allowed molecules are present using :func:`check_only_allowed_molecules`.
        Default = None.
    cutoffs : dict[str, float] | None
        cutoffs of each element. Dictionary with keys for the symbols and values
        of the cutoff radii. If None, use the :data:`ase.data.covalent_radii`. Default: None

    Returns
    -------
    indices : np.ndarray
        Indices required to keep all molecules fully intact.

    """
    to_add: set[int] = set()
    molecules = identify_molecules(atoms, cutoffs=cutoffs)
    if allowed_molecules is not None:
        check_only_allowed_molecules(atoms, molecules, allowed_molecules)

    for molecule in molecules:
        for index in indices:
            if index in molecule:
                to_add.update(molecule)

    to_add.difference_update(indices)
    to_add_list = list(to_add)

    where_to_insert = np.searchsorted(indices, to_add_list)
    indices = np.insert(indices, where_to_insert, to_add_list)

    return indices


def get_permutations_exchange_identical_atoms_groups(
    atoms: Atoms,
    indices: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Get all permutations that result in only swaps of identically bonded atoms.

    Parameters
    ----------
    atoms : Atoms
        Atoms to get permutations for
    indices : Iterable[int] | None
        Default = None.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        All permutations that result in only identical groups swapping

    Notes
    -----
    Perhaps easier to use or more versatile if instead this takes a graph?
    Could then be used directly from the SMILES code, and then not have to generate
    a 3D geometry first.

    Examples
    --------
    >>> # Methanol geometry from CCCBDB
    >>> methanol_positions = [
    ...     [-0.0503, 0.6658, 0.0000],   # C
    ...     [-1.0807, 1.0417, 0.0000],   # H
    ...     [0.4650, 1.0417, 0.8924],    # H
    ...     [0.4650, 1.0417, -0.8924],   # H
    ...     [-0.0503, -0.7585, 0.0000],  # O
    ...     [0.8544, -1.0677, 0.0000],   # H
    ... ]
    >>> methanol = Atoms(symbols="CH3OH", positions=methanol_positions)
    >>> permutations = get_permutations_exchange_identical_atoms_groups(methanol)
    >>> # Should have 6 permutations, only methyl radicals swapping
    >>> len(permutations)
    6
    >>> for permutation in sorted(permutations):
    ...     print(permutation)
    (0, 1, 2, 3, 4, 5)
    (0, 1, 3, 2, 4, 5)
    (0, 2, 1, 3, 4, 5)
    (0, 2, 3, 1, 4, 5)
    (0, 3, 1, 2, 4, 5)
    (0, 3, 2, 1, 4, 5)

    >>> # Only calculate permutations of a subset
    >>> permutations = get_permutations_exchange_identical_atoms_groups(methanol, [0, 1, 2, 5])
    >>> # Should have 2 permutations, only two of the methyl hydrogens
    >>> len(permutations)
    2
    >>> for permutation in sorted(permutations):
    ...     print(permutation)
    (0, 1, 2, 3, 4, 5)
    (0, 2, 1, 3, 4, 5)

    >>> # Ethanol geometry from CCCBDB
    >>> ethanol_positions = [
    ...     [1.1879, -0.3829, 0.0000],   # C
    ...     [2.0985, 0.2306, 0.0000],    # H
    ...     [1.1184, -1.0093, 0.8869],   # H
    ...     [1.1184, -1.0093, -0.8869],  # H
    ...     [0.0000, 0.5526, 0.0000],    # C
    ...     [-0.0227, 1.1812, 0.8852],   # H
    ...     [-0.0227, 1.1812, -0.8852],  # H
    ...     [-1.1867, -0.2472, 0.0000],  # O
    ...     [-1.9237, 0.3850, 0.0000],   # H
    ... ]
    >>> ethanol = Atoms("CH3CH2OH", positions=ethanol_positions)
    >>> permutations = get_permutations_exchange_identical_atoms_groups(ethanol)
    >>> # Should have 12 combinations: 3*2*1 for the CH3 hydrogens, and 2*1 for the CH2 hydrogens
    >>> # So 6*2 = 12 in total.
    >>> len(permutations)
    12

    >>> # Glyoxal geometry from CCCBDB
    >>> glyoxal_positions = [
    ...     [0.0000, 0.7630, 0.0000],   # C
    ...     [0.0000, -0.7630, 0.0000],  # C
    ...     [1.0481, 1.1907, 0.0000],   # H
    ...     [-1.0481, -1.1907, 0.0000], # H
    ...     [-1.0367, 1.3908, 0.0000],  # O
    ...     [1.0367, -1.3908, 0.0000],  # O
    ... ]
    >>> glyoxal = Atoms("C2H2O2", positions=glyoxal_positions)
    >>> permutations = get_permutations_exchange_identical_atoms_groups(glyoxal)
    >>> # Ideally this would give 8 permutations because 2*2*2 identically bonded atoms swapping,
    >>> # but it only gives 2 permutations (the entire molecule rotating 180 degrees).
    >>> len(permutations)
    2
    >>> sorted(permutations)
    [(0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4)]

    """
    graph = _atoms_to_graph(atoms)
    mappings = nx.algorithms.isomorphism.vf2pp_all_isomorphisms(
        graph, graph, node_label="element"
    )
    permutations = []
    for mapping in mappings:
        # If a subset of indices should be permuted, check that all other indices map to themselves
        if indices is not None and not all(
            mapping[i] == i for i in graph.nodes if i not in indices
        ):
            continue
        perm = tuple(mapping[i] for i in range(len(atoms)))
        permutations.append(perm)
    return tuple(perm for perm in permutations)
