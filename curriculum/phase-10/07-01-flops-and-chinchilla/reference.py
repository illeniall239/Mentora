# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def train_flops(n_params: int | float, n_tokens: int | float) -> float:
    if n_params <= 0 or n_tokens <= 0:
        raise ValueError("parameters and tokens must be positive")
    return 6.0 * n_params * n_tokens  # 2 forward + 4 backward FLOPs per parameter per token


def chinchilla_tokens(n_params: int | float) -> float:
    if n_params <= 0:
        raise ValueError("parameters must be positive")
    return 20.0 * n_params
