"""Some utilities to perform ORCA calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.units import Angstrom, fs


def get_calculator_from_orca_inp(
    path: str | Path, profile: OrcaProfile | None = None
) -> ORCA:
    """Create an ORCA instance from an ``orca.inp`` file.

    Parameters
    ----------
    path : str | Path
        Path to ``orca.inp`` file.
    profile : OrcaProfile | None
        OrcaProfile with command to use. Default = None (infer from environment)

    Returns
    -------
    calc : ORCA
        :class:`ase.calculators.orca.ORCA` calculator instance, with its directory,
        charge, multiplicity, orcasimpleinput and orcablocks set.

    """
    path = Path(path)
    with path.open() as file:
        lines = file.readlines()

    calc = _get_calculator_from_orca_inp_lines(lines, profile=profile)
    calc.directory = path.parent
    return calc


def get_calculator_from_orca_out(
    path: str | Path, profile: OrcaProfile | None = None
) -> ORCA:
    """Create an ORCA instance from an ``orca.out`` file.

    Parameters
    ----------
    path : str | Path
        Path to ``orca.out`` file.
    profile : OrcaProfile | None
        OrcaProfile with command to use. Default = None (infer from environment)

    Returns
    -------
    calc : ORCA
        :class:`ase.calculators.orca.ORCA` calculator instance, with its directory,
        charge, multiplicity, orcasimpleinput and orcablocks set.

    Raises
    ------
    RuntimeError
        If ``"INPUT_FILE"`` or ``"****END OF INPUT****"`` are not found in `path`.

    """
    path = Path(path)
    with path.open() as file:
        lines = file.readlines()

    start_line_number, end_line_number = None, None
    for line_idx, line in enumerate(lines):
        if "INPUT FILE" in line:
            start_line_number = line_idx
        elif "****END OF INPUT****" in line:
            end_line_number = line_idx
            break

    if start_line_number is None or end_line_number is None:
        raise RuntimeError

    calc = _get_calculator_from_orca_inp_lines(
        lines[start_line_number:end_line_number], profile=profile
    )
    calc.directory = path.parent
    return calc


def _get_calculator_from_orca_inp_lines(
    lines: list[str], profile: OrcaProfile | None = None
) -> ORCA:
    """Get an ORCA instance from some lines.

    Parameters
    ----------
    lines : list[str]
        lines of, for example, an ``orca.inp`` file.
    profile : OrcaProfile | None
        OrcaProfile with command to use. Default = None (infer from environment)

    Returns
    -------
    calc : ORCA
        :class:`ase.calculators.orca.ORCA` calculator instance, with its charge,
        multiplicity, orcasimpleinput and orcablocks set.

    Raises
    ------
    RuntimeError
        If something goes wrong with the parsing.

    """
    orcablocks = ""
    simpleinput, charge, mult = None, None, None
    for line in lines:
        line = line.split(">")[-1]
        line = line.split("#")[0]
        line = line.strip()
        if not line:
            continue
        if line.startswith("!"):
            simpleinput = line.split("!")[1].strip()
        elif line.startswith("*"):
            split_line = line.split()
            charge = int(split_line[-2])
            mult = int(split_line[-1])
            # The lines after an asterisk contain the geometry, so we can stop
            break
        else:
            orcablocks = "\n".join((orcablocks, line))

    if simpleinput is None or charge is None or mult is None:
        raise RuntimeError

    calc = ORCA(
        profile=profile,
        orcasimpleinput=simpleinput.strip(),
        orcablocks=orcablocks.strip(),
        charge=charge,
        mult=mult,
    )

    return calc


class MDrestart:
    def __init__(self, dct: dict, comment: str | None = None):
        self.dict = dct
        self.comment = comment

    @classmethod
    def from_mdrestart(cls, filepath: str | Path) -> MDrestart:
        with Path(filepath).open("r") as file:
            lines = file.readlines()
        dct = {}
        block_name = None
        for line in lines:
            if line.startswith("#"):
                comment = line
                continue
            if not line.startswith("&"):
                block = "".join([block, line])
                continue
            line = line.strip("&").strip()
            if block_name is not None:
                dct[block_name] = block.rstrip("\n")
            block_name = line
            block = ""
        dct[block_name] = block.rstrip("\n")
        return cls(dct, comment)

    @classmethod
    def from_atoms(
        cls, atoms: Atoms, current_step: int = 0, current_time: float = 0.0
    ) -> MDrestart:
        comment = "# From Atoms instance\n"
        dct = {
            "AtomCount": "   0",
            "CurrentStep": f"   {current_step}",
            "SimulationTime": f"   {current_time}",
            "Positions": "",
            "Velocities": "",
            "Forces": "",
        }
        mdrestart = MDrestart(dct, comment)
        mdrestart.add_atoms(atoms)
        return mdrestart

    def get_atoms(self) -> Atoms:
        symbols = [line.split()[0] for line in self.dict["Positions"].split("\n")]
        positions = (
            np.array(
                [
                    [float(val) for val in line.split()[1:]]
                    for line in self.dict["Positions"].split("\n")
                ]
            )
            / Angstrom
        )
        velocities = (
            np.array(
                [
                    [float(val) for val in line.split()[1:]]
                    for line in self.dict["Velocities"].split("\n")
                ]
            )
            * Angstrom
            / fs
        )
        return Atoms(symbols=symbols, positions=positions, velocities=velocities)

    def write(self, filepath: str | Path) -> None:
        with Path(filepath).open("w") as file:
            if self.comment is not None:
                file.write(self.comment)
            for key, value in self.dict.items():
                file.write(f"&{key}\n")
                file.write(f"{value}\n")

    def add_atoms(self, atoms: Atoms) -> None:
        natoms = len(atoms)
        if "AtomCount" in self.dict:
            self.dict["AtomCount"] = f"   {int(self.dict['AtomCount']) + natoms}"

        if "Positions" in self.dict:
            positions = atoms.positions / Angstrom
            self._add_positions(atoms.symbols, positions)

        if "Velocities" in self.dict:
            velocities = atoms.get_velocities() / (Angstrom / fs)
            self._add_velocities(atoms.symbols, velocities)

        if "Forces" in self.dict:
            forces = np.zeros((natoms, 3))
            self._add_forces(atoms.symbols, forces)

    def _add_positions(self, symbols: list[str], positions: np.ndarray) -> None:
        formatter = {"float": lambda x: f"{x:>25.14f}"}
        for atom_idx, atomic_symbol in enumerate(symbols):
            position_str = np.array2string(
                positions[atom_idx, :],
                formatter=formatter,  # ty: ignore[invalid-argument-type]
                max_line_width=100,
            )[1:-1]
            line = f"   {atomic_symbol} {position_str}"
            if not self.dict["Positions"]:
                self.dict["Positions"] = line
            else:
                self.dict["Positions"] = "\n".join([self.dict["Positions"], line])

    def _add_velocities(self, symbols: list[str], velocities: np.ndarray) -> None:
        formatter = {"float": lambda x: f"{x:>25.14f}"}
        for atom_idx, atomic_symbol in enumerate(symbols):
            velocities_str = np.array2string(
                velocities[atom_idx, :],
                formatter=formatter,  # ty: ignore[invalid-argument-type]
                max_line_width=100,
            )[1:-1]
            line = f"   {atomic_symbol} {velocities_str}"
            if not self.dict["Velocities"]:
                self.dict["Velocities"] = line
            else:
                self.dict["Velocities"] = "\n".join([self.dict["Velocities"], line])

    def _add_forces(self, symbols: list[str], forces: np.ndarray) -> None:
        formatter = {"float": lambda x: f"{x:>25.14f}"}
        for atom_idx, atomic_symbol in enumerate(symbols):
            force_str = np.array2string(
                forces[atom_idx, :],
                formatter=formatter,  # ty: ignore[invalid-argument-type]
                max_line_width=100,
            )[1:-1]
            line = f"   {atomic_symbol} {force_str}"
            if not self.dict["Forces"]:
                self.dict["Forces"] = line
            else:
                self.dict["Forces"] = "\n".join([self.dict["Forces"], line])

    def __repr__(self) -> str:
        return str(self.dict)
