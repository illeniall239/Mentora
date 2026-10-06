# Epsilon-greedy bandit

Topic: 9. Reinforcement learning basics
Difficulty: 1 of 3

## Problem

The simplest reinforcement-learning problem: `n_arms` slot machines, each with an unknown mean reward, and an agent that must earn as much as it can while still finding out which arm is best. Pure Python and `random` only.

`EpsilonGreedy(n_arms: int, eps: float, rng: random.Random)` is an agent with two public attributes, `q` (a list of `n_arms` floats, the running-mean estimate of each arm's reward, all `0.0` at the start) and `counts` (a list of `n_arms` ints, how many times each arm was pulled). Raise `ValueError` if `n_arms < 1` or `eps` is outside `[0, 1]`.

- `select() -> int` draws `u = rng.random()`. If `u < eps`, explore: return `rng.randrange(n_arms)`. Otherwise exploit: return the arm with the highest `q`, the **lowest index** on ties. Use exactly those two `rng` calls in that order so the test can predict your draws.
- `update(arm: int, reward: float) -> None` increments `counts[arm]` and moves `q[arm]` to the mean of every reward that arm has received, using the incremental form `q += (reward − q) / counts` rather than storing the rewards.

The test runs a seeded 4-arm bandit with Gaussian noise for 2 000 steps and expects the `eps = 0.1` agent to pull the best arm at least 75 % of the time. It also runs a purely greedy agent (`eps = 0`) on two arms that pay `0.2` and `0.9` deterministically and expects it to get stuck on the worse arm forever.

## Examples

```
agent = EpsilonGreedy(3, 0.0, random.Random(0))
agent.select()          → 0          all q equal, lowest index wins
agent.update(0, 1.0);  agent.q → [1.0, 0.0, 0.0]
agent.update(0, 0.0);  agent.q → [0.5, 0.0, 0.0],  agent.counts → [2, 0, 0]
agent.update(2, 0.7);  agent.select() → 2

agent = EpsilonGreedy(3, 1.0, random.Random(0))
agent.select()          → rng.randrange(3) after one rng.random() draw: always exploring
```

## Constraints

- `n_arms` at most 16; at most 10 000 steps per run.
- Pure Python and `random` only: no numpy.

## Hints

1. After `k` pulls of an arm with mean estimate `q`, a new reward `r` arrives. Write the new mean in terms of `q`, `r` and `k + 1` without summing all `k + 1` rewards.
2. With `eps = 0.1`, how often does the agent pull a random arm even after it knows the best one? Is that wasted, or does it pay for something?
3. `max(range(n), key=...)` returns the first maximal index. Why does the tie-break matter on the very first pull, when every `q` is `0.0`?
4. A greedy agent pulls arm 0 first, gets `0.2`, and now `q = [0.2, 0.0]`. Which arm does it pull next, and what would ever make it try arm 1?

## Explain-back

- The greedy agent earned `0.2` per step forever while `0.9` was available. Which quantity did it confuse: the reward it saw, or the value of the arms it never tried?
- `q[a]` is an estimate of what? Is it the reward of the last pull, the value of arm `a`, or the return of a policy? How do those three differ?
- Did the agent ever need to know how the bandit generates its rewards? What does that tell you about model-free learning?
- An LLM being aligned by RL sees one reward at the end of a whole response. In what sense is each response an "arm", and why is the space of arms so much harder than four slot machines?
