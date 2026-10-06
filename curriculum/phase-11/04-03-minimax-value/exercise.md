# The optimal discriminator and the minimax value

Topic: 4. GANs, briefly
Difficulty: 3 of 3

## Problem

The GAN paper's central result is about a discriminator you never actually train: for a *fixed* generator, the best possible `D` is known in closed form, and plugging it back into the minimax objective shows what the generator is really minimizing. Work it out numerically on a discrete support, where both distributions are just arrays of probabilities over the same `K` outcomes.

### `optimal_d(p_data: np.ndarray, p_g: np.ndarray) -> np.ndarray`

```
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

elementwise, returning a `float64` array of shape `(K,)`. Where both probabilities are zero the ratio is `0/0`; return `0.5` there (no evidence either way, and the outcome never happens so the choice cannot matter). Every entry lands in `[0, 1]`: it is `1` where only real data appears, `0` where only fakes do, and `0.5` wherever the two agree.

### `minimax_value(p_data: np.ndarray, p_g: np.ndarray) -> float`

The value of the minimax objective at that optimal discriminator:

```
V = sum_x [ p_data(x) * log(D*(x)) + p_g(x) * log(1 - D*(x)) ]
```

Use the convention `0 * log(0) = 0`: a term whose probability weight is zero contributes nothing, even though the logarithm beside it is `-inf`. Return a Python `float`.

`D` maximizes `V` and `G` minimizes it, and this is `V` after `D` has already won its half. Two consequences the tests lean on:

- When `p_g == p_data`, `D*` is `0.5` everywhere and `V = -log 4 = -1.3862943611198906`. That is the **global minimum** over all `p_g`: the generator cannot do better than matching the data, and nothing stops it reaching exactly that number.
- When the supports are disjoint — the generator only ever emits things the data never contains — `D*` is `1` on the data and `0` on the fakes and `V = 0.0`, the largest value possible.

So `V` always lies in `[-log 4, 0]`, and `V + log 4` is twice the Jensen-Shannon divergence between the two distributions.

Both arrays must be 1-D, the same length, non-empty, non-negative, and each must sum to `1` within `1e-8`; raise `ValueError` otherwise. Do not modify the inputs.

## Examples

```
p = array([0.25, 0.25, 0.5])
optimal_d(p, p)                                      → [0.5, 0.5, 0.5]
minimax_value(p, p)                                  → -1.3862943611198906      -log 4

optimal_d(array([1., 0.]), array([0., 1.]))          → [1., 0.]
minimax_value(array([1., 0.]), array([0., 1.]))      → 0.0                      disjoint supports

optimal_d(array([0.5, 0.5, 0.]), array([0.5, 0.5, 0.]))
                                                     → [0.5, 0.5, 0.5]          0/0 is 0.5

optimal_d(array([0.5, 0.5]), array([1., 0.]))        → [0.3333333333333333, 1.]
minimax_value(array([0.5, 0.5]), array([1., 0.]))    → -0.9547712524422192

minimax_value(array([0.5, 0.6]), array([0.5, 0.5]))  → ValueError   (does not sum to 1)
minimax_value(array([1., 0.]), array([1.]))          → ValueError
```

## Constraints

- `K` up to 1024.
- numpy only; no `scipy`, no `np.errstate` trickery needed if you mask properly.
- Compared within `1e-9`.

## Hints

1. `D*(x)` is a ratio of a probability to a sum of probabilities. What is it at an `x` the generator never produces, and what is it at an `x` the data never contains?
2. `0 * log(0)` is `nan` in floating point, not `0`. Which entries can you simply leave out of the sum, and how do you select them without a Python loop?
3. Substitute `D* = 1/2` into `V` by hand, remembering that both distributions sum to 1. How many `log(1/2)` terms survive, and does the answer depend on `K` or on the shape of `p_data`?
4. `V` is reported as the value *after* `D` has been optimized. Which of the two players is still free to move, and does it want this number larger or smaller?

## Explain-back

- `D*` is a ratio of densities. What does that say the discriminator is secretly estimating, and why is that useful even though you throw `D` away at the end?
- Plug `p_g = p_data` into `D*` and then into `V`. Why is `-log 4` the floor and not just one value among many?
- Disjoint supports give `V = 0` and a perfect discriminator. What does that do to the generator's gradient, and which practical GAN failure is this the theory of?
- This whole analysis assumes `D` is optimal for the current `G`. Real training alternates a few steps of each. What does the theory stop guaranteeing once you do that?
