"""Collection of tools to run NEB calculations."""

from copy import deepcopy
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry import conditional_find_mic
from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.mep.neb import NEB, BaseNEB, idpp_interpolate, interpolate
from ase.optimize.lbfgs import LBFGS
from ase.optimize.optimize import Optimizer
from ase.utils.forcecurve import ForceFit, fit_images

from molecular_simulation_tools.utils import check_same_number_of_atoms


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

    Raises
    ------
    ValueError
        If `n_images` is less than 3, because that would result in no intermediate images.

    """
    check_same_number_of_atoms([initial, final])
    if n_images < 3:
        raise ValueError
    images = [initial.copy() for _ in range(n_images - 1)] + [final.copy()]
    return images


def idpp_interpolate_subset(
    images: list[Atoms],
    indices_to_idpp_interp: list[int] | np.ndarray,
    kwargs: dict[str, Any] | None = None,
) -> list[Atoms]:
    """Perform idpp interpolation for a subset, and linear interpolation for the rest.

    Parameters
    ----------
    images : list[Atoms]
        List of images to interpolate between the first and last.
    indices_to_idpp_interp : list[int] | np.ndarray
        Indices to use idpp interpolation for. The rest are interpolated linearly.
    kwargs : dict[str, Any] | None
        Keyword arguments passed to :func:`ase.neb.idpp_interpolate`.
        Default = None.

    Returns
    -------
    interpolated_images : list[Atoms]
        Interpolated images.

    Raises
    ------
    ValueError
        If a key ``"mic"`` is present in `kwargs`. This is inferred from the ``pbc``
        attribute of the images.

    """
    if kwargs is None:
        kwargs = {}
    if "mic" in kwargs:
        raise ValueError

    n_atoms = len(images[0])
    all_indices = np.arange(n_atoms)
    indices_to_linear_interp = all_indices[
        ~np.isin(all_indices, indices_to_idpp_interp)
    ]

    linear_images = [image[indices_to_linear_interp] for image in images]
    interpolate(linear_images)

    idpp_images = [image[indices_to_idpp_interp] for image in images]
    idpp_interpolate(
        idpp_images,
        mic=all(images[0].pbc),
        **kwargs,
    )

    interpolated_images = deepcopy(images)
    for image_idx in range(len(images)):
        interpolated_images[image_idx].positions[indices_to_idpp_interp, :] = (
            idpp_images[image_idx].positions
        )
        interpolated_images[image_idx].positions[indices_to_linear_interp, :] = (
            linear_images[image_idx].positions
        )

    return interpolated_images


def run_energy_weighted_neb(
    images: list[Atoms],
    calc: Calculator,
    fmax: float = 0.05,
    optimizer: type[Optimizer] = LBFGS,
    interpolate: Literal["linear", "idpp"] | None = None,
    neb_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    climb: bool = False,
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
        Keyword arguments passed to :class:`ase.mep.neb.NEB` upon instantiation.
        Default = None.
    optimizer_kwargs : dict[str, Any] | None
        Keyword arguments passed to `optimizer` upon instantiation. Default = None.
    climb : bool
        Whether to do climbing-image (CI) NEB. This will do a two-step NEB, first just a
        regular NEB (converged to 2*fmax), and then turn on the CI-NEB to converge to `fmax`.
        Default = False.

    Returns
    -------
    neb : NEB
        Calculated minimum energy path

    Raises
    ------
    ValueError
        If a key ``"climb"`` is found in `neb_kwargs`. This should be specified directly
        as a keyword argument to this function.

    """
    if neb_kwargs is None:
        neb_kwargs = {}
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if "climb" in neb_kwargs:
        msg = "climb should be specified directly to 'run_energy_weighted_neb', not in 'neb_kwargs'."
        raise ValueError(msg)

    neb = NEB(
        images,
        **neb_kwargs,
    )

    if interpolate is not None:
        print("Interpolating")
        print("Using mic:", all(images[0].pbc))
        neb.interpolate(
            method=interpolate, mic=all(images[0].pbc), apply_constraint=True
        )
        print("Interpolation done")

    for image in neb.images:
        image.calc = calc

    first_neb_fmax = 2 * fmax if climb else fmax
    with optimizer(neb, **optimizer_kwargs) as opt:  # ty: ignore[invalid-argument-type]
        opt.run(fmax=first_neb_fmax)

    if not climb:
        return neb

    images = [image.copy() for image in neb.iterimages()]
    for image in images:
        image.calc = calc

    neb = NEB(images, climb=True, **neb_kwargs)

    with optimizer(neb, **optimizer_kwargs) as opt:  # ty: ignore[invalid-argument-type]
        opt.run(fmax=fmax)

    return neb


def run_zoom_neb(
    images: list[Atoms],
    calc: Calculator,
    fmax: float = 0.1,
    fmax_zoom: float = 0.05,
    energy_weighted_neb_kwargs: dict[str, Any] | None = None,
) -> tuple[NEB, NEB]:
    """Run a ZOOM-NEB calculation.

    A ZOOM-NEB calculation consists of first doing a regular NEB calculation,
    and then a NEB calculation taking the images (in this case) one before and one
    after the TS guess to be the new endpoints.
    See https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/neb.html

    Parameters
    ----------
    images : list[Atoms]
        List of images to find the minimum energy path for.
    calc : Calculator
        Calculator that can calculate the potential energy and forces of the images.
    fmax : float
        Maximum force component criterion on the transition state in eV/Angstrom
        to start the ZOOM. Default = 0.1.
    fmax_zoom : float
        Maximum force component criterion on the transition state in eV/Angstrom.
        Default = 0.1.
    energy_weighted_neb_kwargs : dict[str, Any] | None
        Keyword arguments to pass to :func:`run_energy_weighted_neb`. Default = None.

    Returns
    -------
    neb : NEB
        Calculated minimum energy path
    zoom_neb : NEB
        Calculated zoomed in minimum energy path

    """
    if energy_weighted_neb_kwargs is None:
        energy_weighted_neb_kwargs = {}
    first_neb = run_energy_weighted_neb(
        images,
        calc,
        fmax=fmax,
        **energy_weighted_neb_kwargs,
    )
    indices = range(first_neb.imax - 2, first_neb.imax + 2)
    first_neb_images = list(first_neb.iterimages())

    zoom_neb_initial_images = get_images_for_neb(
        first_neb_images[indices[0]], first_neb_images[indices[-1]], len(images)
    )
    zoom_neb_initial_images[len(images) // 2] = first_neb_images[first_neb.imax].copy()

    zoom_neb = run_energy_weighted_neb(
        zoom_neb_initial_images,
        calc,
        fmax=fmax_zoom,
        **energy_weighted_neb_kwargs,
    )

    return first_neb, zoom_neb


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
    neb = run_energy_weighted_neb(images, calc, **energy_weighted_neb_kwargs)
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
    mask = [False] * (len(images[0]) - 4) + [True] * 4
    with DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        logfile=None,
        mask=mask,
    ) as d_control:
        d_atoms = MinModeAtoms(ts_guess, d_control)

        d_atoms.displace(displacement_vector=displacement_vector)

        # Converge to a saddle point
        with MinModeTranslate(d_atoms, trajectory="ts_opt.traj") as dim_rlx:
            dim_rlx.run(fmax=fmax_ts)
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


def plot_neb(
    images: list[Atoms] | BaseNEB,
    ax: plt.Axes | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> plt.Axes:
    """Plot a NEB calculation.

    Parameters
    ----------
    images : list[Atoms] | BaseNEB
        Images to plot, or NEB instance to get images from
    ax : plt.Axes | None
        Axes to plot on, or None to create a new one. Default = None.
    plot_kwargs : dict[str, Any] | None
        Keyword arguments passed to :meth:`matplotlib.pyplot.Axes.plot`.
        Default = None.

    Returns
    -------
    ax : plt.Axes
        Axes that was plotted on.

    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if ax is None:
        ax = plt.gca()
    if isinstance(images, BaseNEB):
        images = list(images.iterimages())
    force_fit: ForceFit = fit_images(images)
    ax.plot(force_fit.path, force_fit.energies, marker="o", **plot_kwargs)
    return ax
