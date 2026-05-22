"""Collection of geometry tools."""

import itertools
import sys

import numpy as np
from ase import Atoms
from ase.geometry import find_mic


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
