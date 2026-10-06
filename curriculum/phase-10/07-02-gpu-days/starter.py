def gpu_days(flops: float, peak_flops: float, mfu: float, n_gpus: int) -> float:
    """Wall-clock days: flops / (peak_flops * mfu * n_gpus * 86400); ValueError on bad inputs."""
    raise NotImplementedError
