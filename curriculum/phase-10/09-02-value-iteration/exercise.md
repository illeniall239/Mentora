# Value iteration on a tiny MDP

Topic: 9. Reinforcement learning basics
Difficulty: 2 of 3

## Problem

Solve a finite MDP exactly by repeatedly applying the Bellman optimality backup, then read the greedy policy off the solved values. Pure Python: the MDP arrives as plain dicts.

An `mdp` is a dict with four keys:

- `"states"` — a list of hashable state labels, in a fixed order.
- `"terminal"` — a list of the states that end the episode.
- `"actions"` — a dict `state -> list of action labels`, in a fixed order.
- `"transitions"` — a dict `(state, action) -> list of (prob, next_state, reward)` triples whose probabilities sum to 1.

`value_iteration(mdp: dict, gamma: float, tol: float) -> tuple[dict, dict]` returns `(V, policy)`.

- `V` starts as `0.0` for every state in `mdp["states"]`.
- One **sweep** is *synchronous*: every new value is computed from the values at the start of the sweep, never from a value updated earlier in the same sweep. For a non-terminal `s`, `V_new[s] = max over a in mdp["actions"][s] of Q(s, a)` where `Q(s, a) = sum over (p, s2, r) in mdp["transitions"][(s, a)] of p * (r + gamma * V_old[s2])`. Note the reward sits **inside** the expectation and is **not** discounted; only the value of the next state is.
- A **terminal** state keeps `V[s] = 0.0` for ever. Never back up through its actions, even when `mdp["actions"]` and `mdp["transitions"]` list some: the episode is over, so there is nothing after it. The test MDPs include exactly that trap.
- After each sweep compute `delta = max over all states of abs(V_new[s] - V_old[s])` and stop when `delta < tol`, returning the values **from that last sweep**. Stop unconditionally after 10 000 sweeps.
- `policy` is a dict covering the non-terminal states only — terminal states must be absent, not mapped to `None`. Each entry is the action maximising `Q(s, a)` under the final `V`, with ties going to the action that comes **first** in `mdp["actions"][s]`.
- Raise `ValueError` if `gamma` is outside `[0, 1]` or if `tol <= 0`.

Because the stopping rule fires on the first sweep whose `delta` is below `tol`, a loose `tol` makes the returned values depend on the sweep rule: the test runs a three-state chain with `tol = 2.0` and checks the one-sweep synchronous answer, which an in-place (Gauss–Seidel) sweep would not produce. Values are compared with a tolerance of `1e-6`.

## Examples

```
corridor: c0 -R-> c1 -R-> c2 -R-> goal (reward 1.0 on entering goal, 0.0 elsewhere),
          L moves back one cell, L at c0 bumps into the wall; goal is terminal.

value_iteration(corridor, 0.9, 1e-12)
  → V      = {"c0": 0.81, "c1": 0.9, "c2": 1.0, "goal": 0.0}
  → policy = {"c0": "R", "c1": "R", "c2": "R"}        no "goal" key

value_iteration(corridor, 0.5, 1e-12)
  → V["c0"] = 0.25,  V["c1"] = 0.5,  V["c2"] = 1.0    patience is worth less

one state "start" with the single action "go" → (1.0, "goal", 1.0), goal terminal
with a self-loop "stay" → (1.0, "goal", 1.0) that must be ignored:
value_iteration(mdp, 0.9, 1e-12) → V = {"start": 1.0, "goal": 0.0}
                                   (bootstrapping the terminal would give 10.0)

a fork: "risky" → 0.5 of (+10) and 0.5 of (-6), "safe" → +1, all outcomes terminal
value_iteration(mdp, 0.9, 1e-12) → V["fork"] = 2.0, policy["fork"] = "risky"
```

## Constraints

- At most 64 states and 8 actions per state; at most 10 000 sweeps.
- Pure Python: no numpy, no torch.
- Do not mutate the `mdp` argument.

## Hints

1. Write `Q(s, a)` for one state and one action on paper, for a transition list with two outcomes. Which part of it is multiplied by `gamma`, and which part is not?
2. If you overwrite `V[s]` as you walk the state list, what does the next state in the list read when it backs up through `s`? Which `V` does the definition of a sweep say it should read?
3. What is the value of a state the episode has already ended in? What would `V` of that state grow to if you applied the backup to its self-loop anyway, with `gamma = 0.9` and a reward of `1.0` per loop?
4. Once `V` has converged, the policy needs no extra iteration: for each state you compute `Q(s, a)` once per action. Which quantity are you comparing across actions, and why is it not simply the immediate reward?

## Explain-back

- `V["c1"] = 0.9` while every single step in the corridor pays `0.0` until the last one. Which number is the reward here and which is the value, and why are they different?
- The greedy agent at the fork picks `risky`, whose outcome is `-6` half the time. Did it pick the action with the best reward, or the best something else? Name it.
- Your solver read `mdp["transitions"]` on every sweep. An LLM being tuned with policy gradients has no such table. What does that algorithm use instead, and what does it give up?
- The fork pays `+10` or `-6`. If someone rewrote the reward as `+10` for reaching the good outcome *and* `+1` for every step taken on the way, what would the optimal policy start doing, and what is that failure called?
