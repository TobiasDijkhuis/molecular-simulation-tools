"""Tools to identify molecules."""

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
