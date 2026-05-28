"""Collection of functions to calculate properties."""

import numpy as np
from ase import Atoms
from ase.calculators.tip4p import qH
from scipy.fft import fft, fftfreq, next_fast_len
from scipy.signal import correlate, get_window


def get_dipole_moment(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Get the dipole moment of a set of point charges.

    Parameters
    ----------
    positions : np.ndarray
        Positions of point charges. NxM array
    charges : np.ndarray
        Charges. Array of length N

    Returns
    -------
    np.ndarray
        Dipole moment. Array of length M

    Raises
    ------
    ValueError
        If the shapes of `positions` and `charges` do not match.
    ValueError
        If the sum of all `charges` is not 0 (i.e. the structure is not neutral).

    """
    if np.shape(positions)[0] != np.shape(charges)[0]:
        msg = f"Number of position vectors ({np.shape(positions)[0]}) is not the same as the number of charges ({np.shape(charges)[0]})"
        raise ValueError(msg)
    if np.sum(charges) != 0:
        msg = f"Charges should be neutral, but sum of charges is {np.sum(charges):.2f}"
        raise ValueError(msg)

    # dipole_moment = np.zeros(3)
    # for particle_idx in range(np.shape(positions)[0]):
    #     dipole_moment[:] += positions[particle_idx, :] * charges[particle_idx]
    # return dipole_moment
    return np.sum(positions * charges[:, np.newaxis], axis=0)


def get_autocorrelation_function(x: np.ndarray) -> np.ndarray:
    """Get the autocorrelation function of x.

    Parameters
    ----------
    x : np.ndarray
        Array of length N

    Returns
    -------
    np.ndarray
        Array of length N

    """
    return correlate(x, x)[: len(x) + 1]


def get_moving_average(array: np.ndarray, window_size: int) -> np.ndarray:
    """Get the moving average of an array with a certain window size.

    Taken from https://stackoverflow.com/a/14314054.

    Parameters
    ----------
    array : np.ndarray
        array to be smoothed of length ``N``
    window_size : int
        window size

    Returns
    -------
    np.ndarray
        Array of length ``N - window_size + 1``

    """
    array = np.cumsum(array, dtype=float)
    array[window_size:] -= array[:-window_size]
    return array[window_size - 1 :] / window_size


def get_ir_spectrum(
    frames: list[Atoms],
    charges: np.ndarray,
    timestep: float,
    step: int = 1,
    pad: bool = True,
    window_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the infrared spectrum of an MD trajectory.

    Parameters
    ----------
    frames : list[Atoms]
        frames of molecular dynamics simulations
    charges : np.ndarray
        charge of each atom in the Atoms
    timestep : float
        timestep between frames in seconds
    step : int
        Number of steps to take between frames. Default: 1
    pad : bool
        Whether to pad the DACF with zeros to have the FFT be faster.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.next_fast_len.html.
        Default: True
    window_type : str | None
        Window type to apply to get rid of finite
        sampling effects. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window
        for options. Default: None

    Returns
    -------
    xf : np.ndarray
        frequencies in Hz
    spectrum : np.ndarray
        intensity at each frequency. Arbitrary units

    """
    # nsteps = len(frames) // step
    dt = step * timestep

    # dipole_moments = np.zeros(shape=(nsteps, 3))

    positions = np.asarray([frame.positions for frame in frames[::step]])
    velocities = np.gradient(positions, axis=0) * 1e-10 / dt
    charge_weighted_velocities = np.sum(velocities * charges[None, :, None], axis=1)

    # for i, frame in enumerate(frames[::step]):
    #     dipole_moments[i] = get_dipole_moment(frame.positions, charges)

    # dipole_derivatives = np.gradient(dipole_moments, axis=0) / dt

    for i in range(3):
        autocorr = get_autocorrelation_function(charge_weighted_velocities[:, i])

        if i == 0:
            if window_type is not None:
                window = get_window(window_type, len(autocorr))
                autocorr *= window

            if pad:
                padded_length = next_fast_len(len(autocorr))
            else:
                padded_length = len(autocorr)

            spectrum = np.abs(fft(autocorr, n=padded_length)) ** 2
        else:
            spectrum += np.abs(fft(autocorr, n=padded_length)) ** 2
    xf = fftfreq(padded_length, dt)

    # autocorr_function = np.sum(
    #     [get_autocorrelation_function(dipole_derivatives[:, i]) for i in range(3)],
    #     axis=0,
    # )

    # if window_type is not None:
    #     window = get_window(window_type, len(autocorr_function))
    #     autocorr_function *= window

    # if pad:
    #     padded_length = next_fast_len(len(autocorr_function))
    # else:
    #     padded_length = len(autocorr_function)

    # spectrum = np.abs(fft(autocorr_function, n=padded_length))
    # xf = fftfreq(padded_length, dt)

    return xf[: padded_length // 2], spectrum[: padded_length // 2]


def get_h2o_charges(symbols: list[str]) -> np.ndarray:
    """Assign charges of H2O.

    Uses charges from :data:`ase.calculators.tip4p.qH`.

    Parameters
    ----------
    symbols : list[str]
        List of symbols (i.e. "H" or "O")

    Returns
    -------
    charges : np.ndarray
        List of charges corresponding to the atoms in `symbols`.

    Raises
    ------
    ValueError
        If the length of `symbols` is not divisible by 3, or does not have the correct
        composition to be water.

    """
    natoms = len(symbols)

    if natoms % 3 != 0:
        msg = f"Expected number of atoms to be a multiple of 3, but got {natoms} atoms"
        raise ValueError(msg)
    if (
        symbols.count("H") != float(natoms) / 3 * 2
        or symbols.count("O") != float(natoms) / 3
    ):
        msg = "Composition is incorrect for water."
        raise ValueError(msg)

    charges = np.empty(natoms)
    for i, symbol in enumerate(symbols):
        if symbol == "H":
            charges[i] = qH
        elif symbol == "O":
            charges[i] = -2.0 * qH
        else:
            msg = f"Unknown symbol {symbol}"
            raise ValueError(msg)
    return charges
