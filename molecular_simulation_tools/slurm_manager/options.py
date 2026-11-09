"""Managing options for Slurm jobs."""

from pydantic import BaseModel, ConfigDict


class SlurmOptions(BaseModel):
    """Options for a job to be submitted by Slurm."""

    partition: str
    time: str

    model_config = ConfigDict(extra="forbid", frozen=True)
