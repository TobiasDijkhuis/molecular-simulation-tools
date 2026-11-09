"""Collection of tools for input/output."""

from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

from molecular_simulation_tools.eon import read_con_with_info
from molecular_simulation_tools.orca import read_orca_inp
from molecular_simulation_tools.utils import is_number

try:
    import znh5md

    _znh5md_avail = True
except ImportError:
    _znh5md_avail = False


def read(path: str | Path, index: int | str | slice = -1) -> Atoms | list[Atoms]:
    """Read the file based on the filetype.

    Parameters
    ----------
    path : str | Path
        path to file
    index : int | str | slice
        Indices of frames to read. Default = -1 (only last frame).

    Returns
    -------
    Atoms | list[Atoms]
        Read file

    Raises
    ------
    ImportError
        If `path` ends with ``.h5``, but ``znh5md`` is not available.
    NotImplementedError
        When trying to read multiple images from an ``.inp`` file.
    NotImplementedError
        When `index` is not ``":"`` when trying to read a ``.con`` file.

    """
    path = Path(path)
    match path.suffix:
        case ".h5":
            if not _znh5md_avail:
                msg = "Reading h5md files requires znh5md, but it was not available."
                raise ImportError(msg)
            if isinstance(index, str) and index == ":":
                index = slice(None, None, None)
            return znh5md.znh5md.read(path, index=index)  # ty: ignore[invalid-argument-type]
        case ".inp":
            if not _would_read_single_image(index):
                msg = "Reading ORCA input files only supports indexing the one structure in the file."
                raise NotImplementedError(msg)
            return read_orca_inp(path)
        case ".con":
            if index != ":":
                msg = "Single indexing of reading .con files is not implemented. Use index = ':'"
                raise NotImplementedError(msg)
            return read_con_with_info(path)[0]
        case _:
            return ase_read(path, index=index)


def _would_read_single_image(index: int | str | slice) -> bool:
    """Whether the index would read a single image.

    Parameters
    ----------
    index : int | str | slice
        Index to read

    Returns
    -------
    bool
        Whether it would read just a single image.

    """
    return isinstance(index, int) or (isinstance(index, str) and is_number(index))


def read_single(path: str | Path, index: int | str | slice = -1) -> Atoms:
    """Read a single image from a file.

    Just a convenience function for type checkers. See :func:`read`.

    Parameters
    ----------
    path : str | Path
        Path to file.
    index : int | str | slice
        Index to read. Default = -1.

    Returns
    -------
    Atoms
        Read image

    Raises
    ------
    ValueError
        If `index` would cause multiple images to be read.

    """
    if not _would_read_single_image(index):
        msg = f"Index for reading single image '{index}' would not read a single image."
        raise ValueError(msg)
    return read(path, index=index)  # ty: ignore[invalid-return-type]


def read_multiple(path: str | Path, index: int | str | slice = -1) -> list[Atoms]:
    """Read multiple images from a file.

    Just a convenience function for type checkers. See :func:`read`.

    Parameters
    ----------
    path : str | Path
        Path to file.
    index : int | str | slice
        Index to read. Default = -1.

    Returns
    -------
    list[Atoms]
        Read images

    Raises
    ------
    ValueError
        If `index` would cause a single image to be read.

    """
    if _would_read_single_image(index):
        msg = f"Index for reading multiple images '{index}' would not read multiple images."
        raise ValueError(msg)
    return read(path, index=index)  # ty: ignore[invalid-return-type]


def get_number_of_structures_in_xyz(path: str | Path) -> int:
    """Get the number of structures in an ``.xyz`` file.

    Parameters
    ----------
    path : str | Path
        Path to xyz file.

    Returns
    -------
    num_structures : int
        Number of structures in the file.

    """
    lines = Path(path).read_text().split("\n")
    num_lines = len(lines)
    line_number = 0
    num_structures = 0
    while line_number < num_lines:
        line = lines[line_number].strip()
        if line.isdigit():
            num_atoms = int(line_number)
            line_number += num_atoms
            num_structures += 1
        else:
            line_number += 1
    return num_atoms
