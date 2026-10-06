# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
SECONDS_PER_DAY = 86_400


def gpu_days(flops: float, peak_flops: float, mfu: float, n_gpus: int) -> float:
    if flops <= 0 or peak_flops <= 0 or n_gpus <= 0:
        raise ValueError("flops, peak_flops and n_gpus must be positive")
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in (0, 1]")
    useful_flops_per_second = peak_flops * mfu * n_gpus
    return flops / (useful_flops_per_second * SECONDS_PER_DAY)
