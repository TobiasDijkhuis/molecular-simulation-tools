"""Manager for multiple slurm jobs."""

from __future__ import annotations

from molecular_simulation_tools.slurm_manager.job import SlurmJob


class SlurmJobManager:
    """A nice wrapper to manage a bunch of slurm jobs together.

    Parameters
    ----------
    jobs : list[SlurmJob | int | str] | None
        list of jobs to manage, or their ids.

    """

    def __init__(self, jobs: list[SlurmJob | int | str] | None = None):
        self.jobs: list[SlurmJob] = []

        if jobs is None:
            return

        self.add_jobs(jobs)

    def wait(self, sleep_interval: float = 0.5) -> None:
        """Wait until all jobs are done.

        Raises
        ------
        RuntimeError
            If this instance has no jobs.

        """
        if not self.has_jobs():
            msg = "SlurmJobManager has no jobs. Add jobs using SlurmJobManager.add_job or add_jobs first"
            raise RuntimeError(msg)

        for job in self.jobs:
            job.wait(sleep_interval=sleep_interval)

    def cancel(self) -> None:
        """Cancel all jobs.

        Raises
        ------
        RuntimeError
            If this instance has no jobs.

        """
        if not self.has_jobs():
            msg = "SlurmJobManager has no jobs. Add jobs using SlurmJobManager.add_job or add_jobs first"
            raise RuntimeError(msg)
        for job in self.jobs:
            job.cancel()

    def add_job(self, job: SlurmJob | int | str) -> None:
        """Add another job to the jobs to manage.

        Parameters
        ----------
        job : SlurmJob | int | str
            job to manage, or its id

        """
        if isinstance(job, (int, str)):
            # Try to create a SlurmJob from the given id
            job = SlurmJob(job)
        self.jobs.append(job)

    def add_jobs(self, jobs: list[SlurmJob | int | str]) -> None:
        """Add a list of jobs to the jobs to manage.

        Parameters
        ----------
        jobs : list[SlurmJob | int | str]
            list of SlurmJob objects to manage, or their job ids.

        """
        for job in jobs:
            self.add_job(job)

    def has_jobs(self) -> bool:
        """Check whether the instance has jobs attached to it.

        Returns
        -------
        bool
            whether the instance manages job(s)
        """
        return bool(self.jobs)

    def clear_jobs(self) -> None:
        """Clear the list of jobs by setting jobs to be an empty list."""
        self.jobs = []

    def __repr__(self) -> str:
        """Get a printable representation of the manager.

        Returns
        -------
        str
            Nice printable string

        """
        if not self.has_jobs():
            return "SlurmJobManager without jobs"

        val = "SlurmJobManager:"
        for job in self.jobs:
            val += f"\n    {job.__repr__()}"
        return val
