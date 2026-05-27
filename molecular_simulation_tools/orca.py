"""Some utilities to perform ORCA calculations."""

from pathlib import Path

from ase.calculators.orca import ORCA, OrcaProfile


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
