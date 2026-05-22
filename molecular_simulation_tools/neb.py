"""Collection of tools to run NEB calculations."""

from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry import conditional_find_mic
from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.mep.neb import NEB
from ase.optimize.lbfgs import LBFGS
from ase.optimize.optimize import Optimizer


def get_images_for_neb(
    initial: Atoms,
    final: Atoms,
    n_images: int,
) -> list[Atoms]:
    """Get images to use for NEB calculations using :class:`ase.mep.neb.NEB`.

    Creates ``n_images-1`` copies of `initial`, and one copy of `final`.

    Parameters
    ----------
    initial : Atoms
        Starting geometry
    final : Atoms
        Final geometry of the NEB
    n_images : int
        Number of images, including initial and final image

    Returns
    -------
    images : list[Atoms]
        List of images

    """
    images = [initial.copy() for _ in range(n_images - 1)] + [final.copy()]
    return images


def run_energy_weighted_neb(
    images: list[Atoms],
    calc: Calculator,
    fmax: float = 0.05,
    optimizer: type[Optimizer] = LBFGS,
    interpolate: Literal["linear", "idpp"] | None = None,
    neb_kwargs: dict[str, Any] | None = None,
) -> NEB:
    """Do an energy-weighted climbing image nudged elastic band (EW-CI-NEB) calculation.

    See https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00462.

    Parameters
    ----------
    images : list[Atoms]
        List of atoms, from initial to final frame.
    calc : Calculator
        Calculator that can calculate the potential energy and forces of the images.
    fmax : float
        Maximum force on the highest energy component in eV/Angstrom. Default = 0.05.
    optimizer : type[Optimizer]
        Optimizer to use for the NEB. Default = LBFGS.
    interpolate : Literal['linear', 'idpp'] | None
        Method to interpolate see :meth:`ase.mep.neb.NEB.interpolate`.
        If None, do not interpolate images. Default = None.
    neb_kwargs : dict[str, Any] | None
        Keyword arguments passed to :class:`ase.mep.neb.NEB`. Default = None.

    Returns
    -------
    neb : NEB
        Calculated minimum energy path

    """
    if neb_kwargs is None:
        neb_kwargs = {}
    neb = NEB(
        images,
        **neb_kwargs,
    )
    if interpolate is not None:
        neb.images = neb.interpolate(method=interpolate, mic=images[0].pbc)

    for image in neb.images:
        image.calc = calc

    with optimizer(neb, logfile=None) as opt:  # ty: ignore[invalid-argument-type]
        opt.run(fmax=fmax)
        print("EW NEB:", opt.nsteps)
    return neb


def run_neb_ts(
    images: list[Atoms],
    calc: Calculator,
    max_displacement: float = 1e-2,
    fmax_ts: float = 0.05,
    replace_ts_guess: bool = True,
    energy_weighted_neb_kwargs: dict[str, Any] | None = None,
) -> tuple[list[Atoms], Atoms]:
    """Do a NEB-TS calculation.

    NEB-TS calculations consist of first doing an EW-CI-NEB calculation,
    and then a TS optimization on the climbing image.
    See https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00462

    Parameters
    ----------
    images : list[Atoms]
        List of images to find the minimum energy path for.
    calc : Calculator
        Calculator that can calculate the potential energy and forces of the images.
    max_displacement : float
        Maximum displacement of the NEB along the minimum mode in Angstrom.
        Default = 0.02.
    fmax_ts : float
        Maximum force component criterion on the transition state in eV/Angstrom.
        Default = 0.05.
    replace_ts_guess : bool
        Whether to replace the TS guess of the EW-CI-NEB calculation with the
        optimized true transition state. If False, adds it to the list
        in the order according to the rmsd from the initial frame. Default = True.
    energy_weighted_neb_kwargs : dict[str, Any] | None
        Keyword arguments to pass to :func:`run_energy_weighted_neb`. Default = None.

    Returns
    -------
    images : list[Atoms]
        Minimum energy path between the initial and final frame,
        with the transition state optimized.
    transition_state : Atoms
        Optimized transition state

    Raises
    ------
    RuntimeError
        If the neb returned by :func:`run_energy_weighted_neb` does not
        contain energies.

    """
    if energy_weighted_neb_kwargs is None:
        energy_weighted_neb_kwargs = {}
    neb: NEB = run_energy_weighted_neb(images, calc, **energy_weighted_neb_kwargs)
    if neb.energies is None:
        msg = (
            "NEB instance returned by 'run_energy_weighted_neb' does not have energies"
        )
        raise RuntimeError(msg)

    images = list(neb.iterimages())
    max_index = neb.imax
    ts_guess = images[max_index].copy()
    ts_guess.calc = calc

    # Find the image closest in energy to the transition state.
    if neb.energies[neb.imax - 1] > neb.energies[neb.imax + 1]:
        closest_max_index = neb.imax - 1
    else:
        closest_max_index = neb.imax + 1
    closest_to_ts = images[closest_max_index]

    dr, lengths = conditional_find_mic(
        ts_guess.positions - closest_to_ts.positions,
        cell=ts_guess.cell,
        pbc=ts_guess.pbc,
    )
    dr = np.vstack(dr)
    displacement_vector = dr * max_displacement / np.max(lengths)

    # Set up the dimer
    with DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        logfile=None,
        mask=[True for _ in range(len(ts_guess))],
    ) as d_control:
        d_atoms = MinModeAtoms(ts_guess, d_control)

        d_atoms.displace(displacement_vector=displacement_vector)

        # Converge to a saddle point
        with MinModeTranslate(d_atoms, logfile=None) as dim_rlx:
            dim_rlx.run(fmax=fmax_ts)
            print(dim_rlx.nsteps)
        transition_state = d_atoms.get_atoms()

    if replace_ts_guess:
        # Replace TS guess with actual TS
        images[max_index] = transition_state
    else:
        # We need to figure out whether it should be placed before or after the TS,
        # based on the displacement from the reactant image.
        reactant = images[0]
        _, lengths_guess = conditional_find_mic(
            reactant.positions - images[max_index].positions,
            cell=reactant.cell,
            pbc=reactant.pbc,
        )
        _, lengths_ts = conditional_find_mic(
            reactant.positions - transition_state.positions,
            cell=reactant.cell,
            pbc=reactant.pbc,
        )

        rmsd = lambda lengths: np.sqrt(np.average(np.asarray(lengths) ** 2))
        if rmsd(lengths_ts) < rmsd(lengths_guess):
            # TS comes before TS guess
            images.insert(max_index, transition_state)
        else:
            # TS comes after TS guess in path
            images.insert(max_index + 1, transition_state)

    return images, transition_state
