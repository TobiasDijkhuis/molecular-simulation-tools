import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.mep.neb import BaseNEB
from ase.utils.forcecurve import ForceFit, fit_images
from matplotlib import patches


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
    ax.set_xlabel("x (Angstrom)")
    ax.set_ylabel("y (Angstrom)")

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
