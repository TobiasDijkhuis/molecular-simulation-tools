"""A collection of tools to benchmark different ways to calculate the energy."""

import subprocess
from collections.abc import Iterable
from inspect import cleandoc
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db.core import AtomsRow, connect
from ase.db.sqlite import SQLite3Database
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from molecular_simulation_tools.plot import (
    _get_rcparam_value,
    create_secondary_energy_axis,
)
from molecular_simulation_tools.slurm_manager.options import SlurmOptions
from molecular_simulation_tools.utils import is_valid_python_code


def _write_frames(db: SQLite3Database, frames: Iterable[Atoms], **kwargs) -> None:
    with db:
        for frame in frames:
            db.write(frame, **kwargs, **frame.info)


def benchmark(
    calc: Calculator,
    frames: Iterable[Atoms],
    *,
    time: bool = False,
    verbose: bool = False,
) -> list[Atoms]:
    """Run the calculation of `frames` with `calc`.

    Parameters
    ----------
    calc : Calculator
        ASE Calculator that can calculate the energies.
    frames : Iterable[Atoms]
        List of frames to calculate
    time : bool
        Whether to time how long it takes to calculate each frame. Default = False.
    verbose : bool
        Whether to print some information during the calculations. Default = False.

    Returns
    -------
    frames : list[Atoms]
        List of frames with calculators set to
        :class:`ase.calculators.singlepoint.SinglePointCalculator` instances
        with energies set.

    """
    frames = [frame.copy() for frame in frames]
    if time:
        _warmup(calc, frames)
    for frame_idx, frame in enumerate(frames):
        frame.calc = calc
        t = perf_counter()
        energy = frame.get_potential_energy()
        if time:
            frame.info["time_s"] = perf_counter() - t
        frame.info["structure_idx"] = frame_idx
        frame.calc = SinglePointCalculator(frame, energy=energy)  # , forces=forces)
        if verbose:
            print(f"[{calc}] Calculated frame {frame_idx}")
    return frames


def _check_is_database_path(path: str | Path) -> None:
    if not str(path).endswith(".db"):
        raise ValueError


def _warmup(calc: Calculator, frames: Iterable[Atoms]) -> None:
    frames = [frame.copy() for frame in frames]
    for frame in frames:
        frame.calc = calc
        frame.get_potential_energy()
        frame.get_forces()


def benchmark_and_write(
    calc: Calculator,
    frames: Iterable[Atoms],
    method: str,
    db_path: str | Path,
    *,
    time: bool = False,
    verbose: bool = False,
    is_reference: bool = False,
) -> None:
    _check_is_database_path(db_path)

    calculated_frames = benchmark(calc, frames, time=time, verbose=verbose)
    db = connect(db_path)
    _write_frames(db, calculated_frames, method=method, is_reference=is_reference)


def get_all_values(db: SQLite3Database, key: str) -> set[Any]:
    """Get all values of `key` in the database.

    Parameters
    ----------
    db : SQLite3Database
        Database
    key : str
        Key to get values for

    Returns
    -------
    set[Any]
        Set of all values of `key`.

    """
    return {getattr(row, key) for row in db.select() if key in row}


def _get_reference_method(db: SQLite3Database) -> str | None:
    """Get the reference method in a database.

    Parameters
    ----------
    db : SQLite3Database
        Database

    Returns
    -------
    str | None
        String representing the reference method, or None if no rows marked with
        `is_reference=True` were found.

    """
    reference_row = db.get(is_reference=True)
    if reference_row is None:
        return None
    return reference_row.get("method")


def read_benchmark(db: SQLite3Database) -> dict[str, Iterable[AtomsRow]]:
    """Read a benchmarks database, and get the rows per method.

    Parameters
    ----------
    db : SQLite3Database
        Database

    Returns
    -------
    benchmark_results : dict[str, Iterable[AtomsRow]]
        A dictionary of keys the methods and values the rows of the database
        corresponding to that method.

    """
    methods = get_all_values(db, "method")

    benchmark_results: dict[str, Iterable[AtomsRow]] = {}
    for method in methods:
        rows = db.select(method=method, sort="structure_idx")
        benchmark_results[method] = rows
    return benchmark_results


def get_energies_by_method(db: SQLite3Database) -> dict[str, np.ndarray]:
    """Get the energies per method in a benchmarks database.

    Parameters
    ----------
    db : SQLite3Database
        Database

    Returns
    -------
    energies : dict[str, np.ndarray]
        The energies of the structures per method.

    """
    benchmark_results = read_benchmark(db)
    energies: dict[str, np.ndarray] = {}
    for method, results in benchmark_results.items():
        energies[method] = np.asarray([result.get("energy") for result in results])
    return energies


def plot_energies(db: SQLite3Database) -> None:
    """Plot the energies per method in a database.

    Parameters
    ----------
    db : SQLite3Database
        Database

    Notes
    -----
    Only plots the energies where energies are non-zero.
    Makes the method marked with `is_reference=True` black and its linewidth
        twice as thick.

    """
    plt.figure()
    energies = get_energies_by_method(db)
    reference_method = _get_reference_method(db)
    for method in sorted(energies.keys()):
        energies_method = energies[method]
        filtered_indices = np.flatnonzero(energies_method != 0.0)
        energies_filtered = energies_method[filtered_indices]
        plot_kwargs: dict[str, Any] = {"label": method}
        if method == reference_method:
            plot_kwargs.update(
                {"c": "k", "lw": _get_rcparam_value("lines.linewidth") * 2}
            )
        plt.plot(
            filtered_indices,
            energies_filtered - np.min(energies_filtered),
            **plot_kwargs,
        )
    plt.ylabel("Energy (eV)")
    create_secondary_energy_axis(plt.gca(), axis="y")
    plt.legend()
    plt.tight_layout()
    plt.show()


BENCHMARK_FILE_TEMPLATE = """import sys

from ase.io import read
from molecular_simulation_tools.benchmark import benchmark_and_write

{imports}

if __name__ == "__main__":
    inp, out = sys.argv[1], sys.argv[2]
    frames = read(inp, index=":")
    {frame_preprocessing}

    {calculator_definition}

    benchmark_and_write(calc, frames, "{method_identifier}", out, time={time}, verbose={verbose})
"""


class BenchmarkItem(BaseModel):
    """Configuration for a benchmark item."""

    imports: str
    calculator_definition: str
    method: str
    environment: str | None = None
    frame_preprocessing: str = ""
    run_file: str = ""
    is_reference: bool = False
    slurm: SlurmOptions | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def db_path(self) -> str:
        """The path to the database for this item."""
        return self.method + ".db"

    @model_validator(mode="before")
    @classmethod
    def set_run_file(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("run_file") and "method" in data:
            data = {**data, "run_file": f"run_{data['method']}.py"}
        return data

    @field_validator("calculator_definition", mode="after")
    @classmethod
    def check_calculator_definition(cls, v: str) -> str:
        if "calc=" not in v.replace(" ", ""):
            msg = "'calc' variable is never assigned in `calculator_definition`."
            raise ValueError(msg)
        return v

    @field_validator("calculator_definition", "imports", mode="after")
    @classmethod
    def dedent_code(cls, v: str) -> str:
        return cleandoc(v)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark."""

    items: list[BenchmarkItem]
    time: bool
    structures_path: str | Path
    db_path: str | Path

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("items", mode="after")
    @classmethod
    def ensure_items_unique(cls, v: list[BenchmarkItem]) -> list[BenchmarkItem]:
        """Ensure that all items have unique `method` and `run_file`.

        Parameters
        ----------
        v : list[BenchmarkItem]
            The items to run in the benchmark

        Returns
        -------
        v : list[BenchmarkItem]
            The items, unchanged

        Raises
        ------
        ValueError
            If not all the `method` are unique, or not all `run_file` attributes
            are unique.

        """
        methods = [item.method for item in v]
        if len(set(methods)) != len(methods):
            msg = f"Not all methods are unique. Methods:\n{methods}"
            raise ValueError(msg)

        run_files = [str(item.run_file) for item in v]
        if len(set(run_files)) != len(run_files):
            msg = f"Not all run_files are unique. run_files:\n{run_files}"
            raise ValueError(msg)

        return v

    @field_validator("items", mode="after")
    @classmethod
    def ensure_only_one_reference(cls, v: list[BenchmarkItem]) -> list[BenchmarkItem]:
        """Ensure that only one item is marked as `is_reference=True`.

        Parameters
        ----------
        v : list[BenchmarkItem]
            The items to run in the benchmark

        Returns
        -------
        v : list[BenchmarkItem]
            The items, unchanged

        Raises
        ------
        ValueError
            If multiple items are marked with `is_reference=True`.

        """
        references = [item for item in v if item.is_reference]
        if len(references) == 1:
            return v

        msg = f"Multiple items with `is_reference=True` found. Items:\n{references}"
        raise ValueError(msg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Create the BenchmarkConfig from a file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML file.

        Returns
        -------
        BenchmarkConfig
            The instantiated BenchmarkConfig

        """
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


PYTHON_INDENT_STRING = "    "


def _indent_multiline(string: str, indent_level: int = 1) -> str:
    indent_string = "".join(["\n", PYTHON_INDENT_STRING * indent_level])
    return indent_string.join(string.split("\n"))


def get_benchmark_string(
    item: BenchmarkItem, *, time: bool = False, verbose: bool = False
) -> str:
    """Get a string that, when put in a file and ran, will run the item.

    Parameters
    ----------
    item : BenchmarkItem
        Item to run
    time : bool
        Whether to time the benchmark. Default = False.
    verbose : bool
        Whether to print some information during the item run. Default = False.

    Returns
    -------
    str
        The formatted python code.

    """
    dct = {
        "imports": item.imports,
        "calculator_definition": _indent_multiline(item.calculator_definition),
        "method_identifier": item.method,
        "time": time,
        "verbose": verbose,
        "frame_preprocessing": _indent_multiline(item.frame_preprocessing),
    }
    return BENCHMARK_FILE_TEMPLATE.format(**dct)


def create_item_run_file(
    item: BenchmarkItem, *, time: bool = False, verbose: bool = False
) -> None:
    """Create a file that will run a specific BenchmarkItem when run.

    Parameters
    ----------
    item : BenchmarkItem
        Specific item to run in the file
    time : bool
        Whether to time how long it took to run. Default = False.
    verbose : bool
        Whether to print some information during the run. Default = False.

    Raises
    ------
    ValueError
        If the generated python code would result in a SyntaxError.

    """
    file_contents = get_benchmark_string(item, time=time, verbose=verbose)
    if not is_valid_python_code(file_contents):
        msg = f"Item with method {item.method} would generate invalid python in its run_file. run_file:\n{file_contents}"
        raise ValueError(msg)
    Path(item.run_file).write_text(file_contents)


def merge_databases(
    merged_database_path: str | Path, databases: list[str | Path]
) -> None:
    """Merge databases together into one database.

    Parameters
    ----------
    merged_database_path : str | Path
        Merged database
    databases : list[str | Path]
        List of databases to be merged.

    """
    with connect(merged_database_path) as out:
        for path in databases:
            for row in connect(path).select():
                out.write(row, key_value_pairs=row.key_value_pairs, data=row.data)


def setup_benchmark(config: BenchmarkConfig, *, verbose: bool = False) -> None:
    """Write all run_files for the benchmark.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration of the benchmark
    verbose : bool
        Whether to print some information during the run. Default = False.

    """
    for item in config.items:
        create_item_run_file(item, time=config.time, verbose=verbose)


def _get_cmd(item: BenchmarkItem, structures_path: str | Path) -> list[str]:
    """Get the command to run to run a specific item in the benchmark.

    Parameters
    ----------
    item : BenchmarkItem
        Item to run
    structures_path : str | Path
        Path to the structures to run in benchmark.

    Returns
    -------
    list[str]
        Command to run.

    """
    cmd = ["python3", item.run_file, str(structures_path), item.db_path]
    if item.environment is not None:
        cmd = ["conda", "run", "-n", str(item.environment), *cmd]
    return cmd


def run_benchmark(config: BenchmarkConfig) -> None:
    """Run a benchmark, and merge the databases of each item.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration of the benchmark.

    Raises
    ------
    RuntimeError
        If one of the items failed.

    """
    procs = [
        subprocess.Popen(_get_cmd(item, config.structures_path), text=True)
        for item in config.items
    ]
    for p in procs:
        if p.wait() != 0:
            msg = f"Run failed: {p.args}"
            raise RuntimeError(msg)
    merge_databases(config.db_path, [item.db_path for item in config.items])
