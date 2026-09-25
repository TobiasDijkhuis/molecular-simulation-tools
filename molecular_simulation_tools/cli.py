from pathlib import Path
from typing import Annotated

import typer
from ase.visualize import view as ase_view

from molecular_simulation_tools.io import read

app = typer.Typer()


@app.command()
def gui(
    path: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
    block: Annotated[bool, typer.Option()] = False,
) -> None:
    """View a molecular geometry. Reads more filetypes than 'ase gui'"""
    ase_view(read(path), block=block)


@app.command()
def null() -> None:
    """Do nothing"""
    pass


if __name__ == "__main__":
    app()
