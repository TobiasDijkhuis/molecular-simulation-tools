"""Collection of tools for input/output."""

from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

from molecular_simulation_tools.eon import read_con_with_info
from molecular_simulation_tools.orca import read_orca_inp

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
            if index != -1:
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
