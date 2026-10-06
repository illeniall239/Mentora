# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
MAX_SWEEPS = 10_000


def _q(mdp: dict, state, action, gamma: float, V: dict) -> float:
    # reward is inside the expectation and undiscounted; only the next state's value is discounted
    return sum(p * (r + gamma * V[s2]) for p, s2, r in mdp["transitions"][(state, action)])


def value_iteration(mdp: dict, gamma: float, tol: float) -> tuple[dict, dict]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if tol <= 0:
        raise ValueError("tol must be positive")

    terminal = set(mdp["terminal"])
    live = [s for s in mdp["states"] if s not in terminal]
    V = {s: 0.0 for s in mdp["states"]}

    for _ in range(MAX_SWEEPS):
        old = V  # every backup in this sweep reads the values from the start of the sweep
        V = dict(old)
        for s in live:
            V[s] = max(_q(mdp, s, a, gamma, old) for a in mdp["actions"][s])
        if max(abs(V[s] - old[s]) for s in V) < tol:
            break

    policy = {}
    for s in live:
        actions = mdp["actions"][s]
        best = max(range(len(actions)), key=lambda i: (_q(mdp, s, actions[i], gamma, V), -i))
        policy[s] = actions[best]  # ties go to the first action listed
    return V, policy
