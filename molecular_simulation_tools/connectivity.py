"""Tools to identify molecules. Taken from IPSuite."""

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

    Requires networkx to be installed.

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

    Requires networkx to be installed.

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
