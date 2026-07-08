"""Some utilities for managing slurm jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
from subprocess import run


def is_slurm_active() -> bool:
    """Run a command to find out if slurm is installed and active.

    Returns
    -------
    bool
        whether slurm is installed and active.

    """
    try:
        command = run(["sacct", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return not command.stderr


def get_user() -> str:
    """Get the username of the user by running "$whoami".

    Returns
    -------
    str
        user

    """
    return run(["whoami"], capture_output=True, text=True).stdout


def same_submission_script_already_submitted(
    submission_script_path: str | Path,
) -> tuple[bool, int | None]:
    """Whether the same submission script has already been submitted to Slurm.

    Inputs
    ------
    submission_script_path : str | Path
        path to job submission script. Absolute path.

    Returns
    -------
    tuple[bool, int | None]
        Whether the job has already been submitted, and if so its job ID.

    Raises
    ------
    RuntimeError
        If ...

    """
    command = f"squeue -u {get_user()} --format=%A,%o,%T,%Z"
    output = run(command.split(), capture_output=True, text=True).stdout
    if str(submission_script_path) not in output:
        return (False, None)
    jobs_info = output.split("\n")
    for job_info in jobs_info:
        if str(submission_script_path) in job_info:
            job_id = int(job_info.split(",")[0])
            return (True, job_id)
    raise RuntimeError
