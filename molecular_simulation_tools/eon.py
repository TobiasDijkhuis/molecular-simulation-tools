"""Helper functions to run calculations with eOn v2."""

from __future__ import annotations

import configparser
import csv
import inspect
import re
import subprocess
import warnings
from dataclasses import dataclass
from enum import StrEnum, auto, unique
from json import loads
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, get_type_hints

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.eon import read_eon, write_eon

from molecular_simulation_tools.utils import (
    capitalize_each_word,
    get_directory,
    set_current_directory,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_EON_JOB_REQUIRED_KEYS = frozenset({"Main", "Potential"})

_EON_ASE_CALC_INTERFACE = """def _calculate(R, atomicNrs, box, calc):  # ruff: ignore[missing-type-function-argument, invalid-argument-name]
    system = Atoms(symbols=atomicNrs, positions=R, pbc=True, cell=box)
    system.calc = calc
    forces = system.get_forces()
    energy = system.get_potential_energy()
    return energy, forces
"""

EON_ASE_DUMMY_CALC_INTERFACE = dedent("""    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes


    class DummyCalculator(Calculator):
        implemented_properties = [
            "energy",
            "forces",
        ]

        def __init__(self, **kwargs):
            Calculator.__init__(self, **kwargs)

        def calculate(  # ty: ignore[invalid-method-override]
            self,
            atoms: Atoms,
            properties: list[str] = ["energy"],
            system_changes: list[str] = all_changes,
        ) -> None:
            Calculator.calculate(self, atoms, properties, system_changes)
            self.results["energy"] = 0
            self.results["forces"] = np.zeros_like(atoms.positions)


    def ase_calc() -> DummyCalculator:
        return DummyCalculator()""")


def _verify_calc_function(ase_calc: Callable[[], Calculator]) -> None:
    """Verify that a function is valid for eOn v2 to use as an ASE calculator.

    Parameters
    ----------
    ase_calc : Callable[[], Calculator]
        Function that initializes an ASE Calculator

    Raises
    ------
    TypeError
        If `ase_calc` is not a function.
    ValueError
        If `ase_calc` takes arguments.
    NameError
        If `ase_calc` is not named ``ase_calc``.
    ValueError
        If `ase_calc` does not return a subclass of
        :class:`ase.calculators.calculator.Calculator"

    """
    if not inspect.isfunction(ase_calc):
        msg = "The ase_calc function is not a function."
        raise TypeError(msg)

    sig = inspect.signature(ase_calc)
    if sig.parameters:
        msg = f"The ase_calc function should not take any arguments, but found {sig.parameters}"
        raise ValueError(msg)

    if ase_calc.__name__ != "ase_calc":
        msg = f"ase_calc function should be called 'ase_calc', but was called '{ase_calc.__name__}'"
        raise NameError(msg)

    hints = get_type_hints(ase_calc)
    return_annotation = hints.get("return")
    if return_annotation is None:
        msg = f"Could not infer the return type of {ase_calc.__name__}. Add return type annotation"
        warnings.warn(msg, category=SyntaxWarning, stacklevel=1)
        return
    if not issubclass(return_annotation, Calculator):
        msg = f"Returned class by {ase_calc.__name__} is not a subclass of ASE Calculator."
        raise ValueError(msg)


def get_calc_string(ase_calc: Callable[[], Calculator]) -> str:
    """Get the code from a function that initializes an ASE calculator.

    Parameters
    ----------
    ase_calc : Callable[[], Calculator]
        Function called ``ase_calc`` that initializes an ASE Calculator.

    Returns
    -------
    str
        String representation of the code of `ase_calc`.

    """
    _verify_calc_function(ase_calc)
    return inspect.getsource(ase_calc)


def write_calc_string(string: str, path: str | Path) -> None:
    """Write a file string that defines how to initialze an ASE calculator.

    Parameters
    ----------
    string : str
        String to write. This must contain the ``def ase_calc`` that eOn v2 uses
        to initialize the calculator.
    path : str | Path
        Path to write to.

    Raises
    ------
    ValueError
        If :class:`ase.Atoms` is not imported in `string`, or ``def ase_calc`` is not found
        in `string`.

    """
    if "from ase import Atoms" not in string:
        msg = "Atoms was not imported in setup. Add 'from ase import Atoms'."
        raise ValueError(msg)

    if "def ase_calc" not in string:
        msg = "'def ase_calc' was not found in string, so EON would not be able to initialize the calculator."
        raise ValueError(msg)
    Path(path).write_text("\n\n\n".join((string.strip(), _EON_ASE_CALC_INTERFACE)))


def write_eon_structure(atoms: Atoms | list[Atoms], path: str | Path) -> None:
    if isinstance(atoms, Atoms):
        atoms: list[Atoms] = [atoms]
    path = Path(path)

    if path.is_file():
        path.unlink()

    with path.open(mode="w") as file:
        write_eon(file, atoms)


def _check_and_convert_eon_dict(dct: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked_dct = {}
    for key, value in dct.items():
        for subkey, subvalue in value.items():
            if isinstance(subvalue, bool):
                value[subkey] = str(subvalue).lower()
        checked_dct[capitalize_each_word(key)] = value

    for key in _EON_JOB_REQUIRED_KEYS:
        if key not in checked_dct:
            msg = f"Missing key '{key}' in checked_dct. Available keys: {checked_dct.keys()}"
            raise KeyError(msg)
    return checked_dct


def write_eon_job(dct: dict[str, Any], path: str | Path) -> None:
    """Write a job for eOn v2.

    Parameters
    ----------
    dct : dict[str, Any]
        Dictionary for the job.
    path : str | Path
        Path to write to.

    """
    dct = _check_and_convert_eon_dict(dct)
    parser = configparser.ConfigParser()
    parser.update(dct)
    with Path(path).open(mode="w") as file:
        parser.write(file)


def read_con_info(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(mode="r") as file:
        info_dicts = [
            loads(line)
            for line in file
            if line.startswith("{") and line.endswith("}\n")
        ]
    return info_dicts


def read_con_with_info(path: str | Path) -> tuple[list[Atoms], list[dict[str, Any]]]:
    atoms: list[Atoms] = read_eon(path)
    info_dicts = read_con_info(path)
    for atom, info_dict in zip(atoms, info_dicts, strict=True):
        if "energy" not in info_dict:
            raise ValueError
        atom.calc = SinglePointCalculator(atom, energy=info_dict["energy"])
    return atoms, info_dicts


OneDIntArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]
OneDFloatArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]


class NEBData(NamedTuple):
    """Data about the NEB path."""

    img: OneDIntArray
    rxn_coord: OneDFloatArray
    energy: OneDFloatArray
    f_para: OneDFloatArray


def read_neb_data(filepath: str | Path) -> NEBData:
    """Read a ``"neb.data"`` file.

    Parameters
    ----------
    filepath : str | Path
        Path to the file.

    Returns
    -------
    NEBData
        Data about the NEB path.

    """
    with Path(filepath).open() as file:
        reader = csv.reader(file, delimiter=" ", skipinitialspace=True)

        dct: dict[str, list[float]] = {key: [] for key in next(reader)}
        keys = list(dct.keys())
        for row in reader:
            for col, value in enumerate(row):
                dct[keys[col]].append(float(value))
    img = np.array(dct.pop("img"), dtype=int)
    return NEBData(img=img, **{key: np.array(value) for key, value in dct.items()})


_RE_NEB_CONVERGED = re.compile(r"^NEB converged")
_RE_NEB_FAILED = re.compile(r"^Nudged elastic band, too many iterations.")
_RE_NEB_INTERRUPTED = re.compile("ASE calculator Python error: KeyboardInterrupt")
_RE_POTENTIAL_CALLS = re.compile(r"called potential ([0-9]+) times")
EON_LOG_FILEPATH = "client_quill.log"


@unique
class NEBStatus(StrEnum):
    """Status of the NEB."""

    CONVERGED = auto()
    TOO_MANY_ITERATIONS = auto()
    RUNNING = auto()
    INTERRUPTED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, kw_only=True)
class NEBInfo:
    """Info about the NEB run."""

    status: NEBStatus
    num_evaluations: int
    num_iterations: int
    force_convergence: OneDFloatArray

    def converged(self) -> bool:
        """Whether the NEB job converged.

        Returns
        -------
        bool
            Whether the NEB converged.

        """
        return self.status == NEBStatus.CONVERGED


def _read_iteration_info_from_lines(lines: list[str]) -> tuple[int, OneDFloatArray]:
    re_optimization_start = re.compile(r"iteration\s+step size")
    start_idx = None
    end_idx = None
    for line_nr, line in enumerate(lines):
        if re_optimization_start.search(line) is not None:
            start_idx = line_nr + 2
        elif get_status_from_line(line) is not None:
            end_idx = line_nr

    if start_idx is None:
        joined_lines = "\n".join(lines)
        msg = f"Could not infer starting index of optimization from lines. Lines:\n{joined_lines}"
        raise RuntimeError(msg)

    iteration_lines = [
        line.split()
        for line in lines[start_idx:end_idx]
        if line and line.split()[0].isnumeric()
    ]
    force_convergence = np.array(
        [float(split_line[2]) for split_line in iteration_lines]
    )
    num_iterations = len(iteration_lines)
    return num_iterations, force_convergence


def get_status_from_line(line: str) -> NEBStatus | None:
    """Get the status of the NEB from a line in a log file.

    Parameters
    ----------
    line : str
        Line in :data:`EON_LOG_FILEPATH` file.

    Returns
    -------
    NEBStatus | None
        Status, or None if it could not be determined from this line.

    """
    m = _RE_NEB_CONVERGED.search(line)
    if m is not None:
        return NEBStatus.CONVERGED
    m = _RE_NEB_FAILED.search(line)
    if m is not None:
        return NEBStatus.TOO_MANY_ITERATIONS
    m = _RE_NEB_INTERRUPTED.search(line)
    if m is not None:
        return NEBStatus.INTERRUPTED


def read_eon_quill_log(path: str | Path) -> NEBInfo:
    """Read the eOn v2 quill log.

    Parameters
    ----------
    path : str | Path
        Path to the logfile.

    Returns
    -------
    NEBInfo
        Info about the NEB run

    """
    num_potential_evaluations = 0
    status = NEBStatus.UNKNOWN

    lines = Path(path).read_text().split("\n")
    for line in lines:
        m = _RE_POTENTIAL_CALLS.search(line)
        if m is not None:
            num_potential_evaluations += int(m.group(1))
            continue

        if status != NEBStatus.UNKNOWN:
            continue
        new_status = get_status_from_line(line)
        if new_status is not None:
            status = new_status

    num_iterations, force_convergence = _read_iteration_info_from_lines(lines)

    return NEBInfo(
        num_evaluations=num_potential_evaluations,
        status=status,
        num_iterations=num_iterations,
        force_convergence=force_convergence,
    )


def run_eon_job(path: str | Path) -> subprocess.CompletedProcess:
    """Run the eOn job in a certain directory.

    Parameters
    ----------
    path : str | Path
        Path to the directory or eOn input file.

    Returns
    -------
    result : subprocess.CompletedProcess
        Process that was run.

    """
    with set_current_directory(get_directory(path)):
        result = subprocess.run(
            ["eonclient"],
            capture_output=True,
            text=True,
            check=True,
        )
    return result
