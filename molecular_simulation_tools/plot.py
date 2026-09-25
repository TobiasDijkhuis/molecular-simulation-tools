"""Tools for creating plots."""

import itertools
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.mep.neb import BaseNEB
from ase.units import eV, kcal, mol
from ase.utils.forcecurve import ForceFit, fit_images
from matplotlib import patches
from matplotlib.axes._secondary_axes import SecondaryAxis


def plot_neb_from_images(
    images: list[Atoms] | BaseNEB,
    ax: plt.Axes | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    *,
    mark_transition: bool = True,
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
    if isinstance(images, BaseNEB):
        images = list(images.iterimages())
    force_fit: ForceFit = fit_images(images)
    return plot_neb(
        force_fit.path,
        force_fit.energies,
        ax=ax,
        plot_kwargs=plot_kwargs,
        mark_transition=mark_transition,
    )


def plot_neb(
    path: np.ndarray,
    energies: np.ndarray,
    ax: plt.Axes | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    *,
    mark_transition: bool = False,
    include_secondary_axis: bool = True,
) -> plt.Axes:
    if plot_kwargs is None:
        plot_kwargs = {}
    if ax is None:
        ax = plt.gca()

    relative_energies = energies - energies[0]
    ax.plot(path, relative_energies, marker="o", **plot_kwargs)
    if mark_transition:
        ts_index = np.argmax(relative_energies)
        ax.scatter(
            path[ts_index],
            relative_energies[ts_index],
            marker="*",
            s=200,
            color="yellow",
            edgecolor="black",
            zorder=5,
        )

    ax.set_xlabel(r"Reaction coordinate ($\mathrm{\AA}$)")
    ax.set_ylabel("Energy (eV)")
    if include_secondary_axis:
        create_secondary_energy_axis(ax)
    return ax


def create_secondary_energy_axis(
    ax: plt.Axes, axis: Literal["x", "y"] = "y"
) -> tuple[plt.Axes, SecondaryAxis]:
    """Create a secondary energy axis in kcal/mol.

    Parameters
    ----------
    ax : plt.Axes
        Axes object to modify
    axis : Literal['x', 'y']
        Which axis to add the secondary axis to. Default = 'y'

    Returns
    -------
    ax : plt.Axes
        Original Axes object.
    secax : SecondaryAxis
        Newly created secondary axis.

    Raises
    ------
    ValueError
        If `axis` is not ``"x"`` or ``"y"``.

    """
    ev_to_kcal_per_mol = lambda energy_ev: energy_ev * eV / (kcal / mol)
    kcal_per_mol_to_ev = lambda energy_kcal_per_mol: (
        energy_kcal_per_mol * (kcal / mol) / eV
    )

    if axis == "y":
        secax = ax.secondary_yaxis("right", (ev_to_kcal_per_mol, kcal_per_mol_to_ev))
        ax.tick_params(axis="y", right=False, which="both")
        secax.set_ylabel("Energy (kcal/mol)")
    elif axis == "x":
        secax = ax.secondary_xaxis("top", (ev_to_kcal_per_mol, kcal_per_mol_to_ev))
        ax.tick_params(axis="x", top=False, which="both")
        secax.set_xlabel("Energy (kcal/mol)")
    else:
        msg = f"'axis' should be one of ['x', 'y'], but was '{axis}'"
        raise ValueError(msg)
    return ax, secax


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
    ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")

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

    # TODO: Better inference of which periodic images are necessary.
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
