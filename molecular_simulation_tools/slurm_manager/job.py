"""A nice way to manage a slurm job."""

from __future__ import annotations

import os
from pathlib import Path
from subprocess import run
from time import sleep

from molecular_simulation_tools.slurm_manager.utils import (
    get_user,
    same_submission_script_already_submitted,
)


class SlurmJob:
    """Class representing a slurm job.

    Parameters
    ----------
    job_id : str | int
        Job ID returned after slurm job is started.

    """

    def __init__(self, job_id: str | int):
        if not isinstance(job_id, (str, int)):
            msg = f"job_id should be type int or string, but was type {type(job_id)}"
            raise TypeError(msg)

        try:
            self.job_id: int = int(job_id)
        except ValueError as e:
            msg = f"Could not convert job_id {job_id} to an integer, which means that the job id is not valid."
            raise ValueError(msg) from e

        self._check_in_queue()

    @classmethod
    def start_from_command(
        cls, command: str, directory: str | Path = ".", force: bool = False
    ) -> SlurmJob:
        """Start a slurm job from a command.

        Start a job from a command in a certain directory,
        and initialize a SlurmJob instance from the resulting job ID.

        Parameters
        ----------
        command : str
            Command to execute.
            Should probably be something with "sbatch" and a shell script.
        directory : str | Path
            Directory in which to execute command.
            The directory should already exist. Default: "."

        Returns
        -------
        SlurmJob
            Initialized `SlurmJob` instance.

        Raises
        ------
        ValueError
            If the command does not contain a file as its second argument.
        FileNotFoundError
            If the executable cannot be found.
        RuntimeError
            If there was an error starting the job.
        ValueError
            If the job ID could not be extracted from the return message of starting
            the job.

        """
        init_dir = Path.cwd()

        # Change to correct directory
        os.chdir(directory)

        command_split = command.split()
        if not Path(command_split[1]).is_file():
            msg = f"Command {command} does not contain file a file {command_split[1]}"
            raise ValueError(msg)

        submission_script_path = Path(directory) / command_split[1]
        submitted, job_id = same_submission_script_already_submitted(
            submission_script_path
        )
        if submitted and not force:
            print(
                f"Detected that job with submission command {command} has already started, with job ID {job_id}. Not submitting again"
            )
            os.chdir(init_dir)
            return cls(job_id)  # type: ignore[arg-type, ty:invalid-argument-type]

        # Get job id from this command
        try:
            process = run(command.split(), capture_output=True, text=True)
        except FileNotFoundError as e:
            msg = f"Could not execute command {command} because the executable {command.split(maxsplit=1)[0]} could not be found"
            raise FileNotFoundError(msg) from e

        if process.stderr:
            msg = f"Error starting job: {process.stderr}"
            raise RuntimeError(msg)

        try:
            job_id = int(process.stdout.split()[-1])
        except ValueError as e:
            msg = f"Could not extract job_id from stdout {process.stdout}"
            raise ValueError(msg) from e

        os.chdir(init_dir)

        return cls(job_id)

    def get_state(self) -> str:
        """Get the state of the job.

        Returns
        -------
        state : str
            State of the job.

        """
        # Sacct command here
        command = f"sacct --noheader --jobs {self.job_id} --format=jobid,state"

        stdout = run(command.split(), capture_output=True, text=True).stdout
        while not stdout:
            # It is possible that sacct has some trouble finding the job.
            # If that is the case, wait for a bit, and then try again.
            sleep(0.1)
            stdout = run(command.split(), capture_output=True, text=True).stdout
        state = stdout.split("\n")[0].split()[1]
        return state

    def is_completed(self) -> bool:
        """Whether the current status of the job is "COMPLETED".

        Returns
        -------
        bool:
            Whether the current status of the job is "COMPLETED"

        """
        return self.get_state() == "COMPLETED"

    def is_done(self) -> bool:
        """Check whether the job is done.

        Done means not the state is not "RUNNING" or "PENDING".
        A job failing (with status "FAILED") also means that it is done.

        Returns
        -------
        bool
            Whether the job is done.

        """
        return self.get_state() not in {"RUNNING", "PENDING"}

    def wait(self, sleep_interval: float = 0.5) -> None:
        """Wait until the job is done.

        Parameters
        ----------
        sleep_interval : float
            Time between checking the state in seconds.

        """
        # wait until the calculation is done.
        while not self.is_done():
            sleep(sleep_interval)

    def cancel(self) -> None:
        """Cancel this job."""
        run(["scancel", str(self.job_id)])

    def __repr__(self) -> str:
        """Get nice printable representation of the job.

        Returns
        -------
        str
            Nice printable string

        """
        return f"SlurmJob: ID {self.job_id}, state {self.get_state()}"

    def __str__(self) -> str:
        """Get string representation of the job.

        Returns
        -------
        str :
            Stringified job id.

        """
        return str(self.job_id)

    def _check_in_queue(self) -> None:
        """Check that the job is in the queue, and if not raises an error.

        Raises
        ------
        RuntimeError
            If the jobs job id is not valid.

        """
        # Function that runs that makes sure this job id is in the queue.
        # This print the job and its state. If a stderr is thrown,
        # the job is not in the queue and it is not a valid job id,
        # we raise a ValueError
        command = (
            f"squeue --user {get_user()} --format=%i,%T --jobs {self.job_id} --noheader"
        )
        process = run(command.split(), capture_output=True, text=True)
        if process.stderr:
            msg = f"Job ID {self.job_id} is not valid: {process.stderr}"
            raise RuntimeError(msg)
