# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math

import torch


def incremental_attention(
    cache: dict[str, torch.Tensor], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K = torch.cat([cache["K"], k.unsqueeze(0)], dim=0)
    V = torch.cat([cache["V"], v.unsqueeze(0)], dim=0)
    scores = K @ q / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=0)
    return weights @ V, {"K": K, "V": V}
