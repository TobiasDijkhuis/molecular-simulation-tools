"""Collection of tools for input/output."""

from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

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
        If `path` ends with `.h5`, but ``znh5md`` is not available.

    """
    if str(path).endswith(".h5"):
        if not _znh5md_avail:
            msg = "Reading h5md files requires znh5md, but it was not available."
            raise ImportError(msg)
        if isinstance(index, str) and index == ":":
            index = slice(None, None, None)
        return znh5md.znh5md.read(path, index=index)  # ty: ignore[invalid-argument-type]
    return ase_read(path, index=index)
