from pathlib import Path

from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile

from molecular_simulation_tools.orca import get_calculator_from_orca_inp

orca_test_calculator = ORCA(
    profile=OrcaProfile("echo 'dummy'"),
    orcasimpleinput="wB97M-D4 def2-TZVPD EnGrad TightSCF",
    charge=0,
    mult=2,
    orcablocks="% HFTyp UHF end\n% Guess PModel end",
)
test_atoms = Atoms(symbols="H", positions=[[0, 0, 0]])


def test_get_calculator_from_orca_inp(tmpdir):
    dir = Path(tmpdir)
    orca_test_calculator.directory = dir
    orca_test_calculator.write_inputfiles(test_atoms, properties=["energy"])
    calculator = get_calculator_from_orca_inp(
        dir / "orca.inp", profile=OrcaProfile("echo 'dummy'")
    )
    assert calculator.parameters == orca_test_calculator.parameters
    assert calculator.directory == orca_test_calculator.directory
