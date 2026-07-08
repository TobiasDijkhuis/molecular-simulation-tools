from . import job, manager, utils

if not utils.is_slurm_active():
    msg = f"Tried to import {__name__}, but slurm is not installed or not active"
    raise ImportError(msg)
