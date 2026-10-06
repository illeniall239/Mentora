# One REINFORCE step

Topic: 9. Reinforcement learning basics
Difficulty: 3 of 3

## Problem

Compute the REINFORCE policy gradient for a softmax policy by hand, in numpy, and check it against a numeric gradient. This is the whole idea RLHF is built on: push up the log-probability of the actions that led to a high return, push down the ones that did not.

The policy is `pi(a | s) = softmax(logits[s])[a]`. A batch of `n` sampled steps arrives as:

- `logits` — float array of shape `(n, A)`, one row of action logits per sampled step.
- `actions` — int array of shape `(n,)`, the action actually sampled at each step, in `[0, A)`.
- `returns` — float array of shape `(n,)`, the return that followed each step.
- `baseline` — a float subtracted from every return.

`reinforce_grad(logits, actions, returns, baseline=0.0) -> np.ndarray` returns an array of shape `(n, A)`:

```
J = (1 / n) * sum_i (returns[i] - baseline) * log pi(actions[i] | logits[i])
grad[i, j] = d J / d logits[i, j] = (returns[i] - baseline) * (1[j == actions[i]] - p[i, j]) / n
```

where `p = softmax(logits)` row-wise.

- This is the gradient of the **objective**, the ascent direction: a training loop does `logits += lr * grad`. It is not the gradient of a loss, so do not negate it. A positive advantage must make `grad[i, actions[i]]` positive.
- The advantage `returns[i] - baseline` multiplies the whole `(1[j == a] - p)` vector. The baseline is subtracted from the return, never from a probability or a log-probability, and it changes the variance of the estimate, not what it estimates on average.
- Divide by `n`: the result is a mean over the batch, not a sum.
- Compute the softmax row-wise after subtracting each row's maximum, so a row like `[1000.0, 1001.0]` does not overflow.
- Raise `ValueError` if `logits` is not 2-D, if `n == 0`, if `actions` or `returns` do not have length `n`, or if any action is outside `[0, A)`.
- numpy only: no torch, and no autograd of any kind — write the derivative yourself. Tolerance for all comparisons is `1e-6`.

## Examples

```
reinforce_grad([[0.0, 0.0]], [0], [1.0])
  → [[0.5, -0.5]]                      p = [0.5, 0.5], advantage 1, one sample

reinforce_grad([[0.0, 0.0], [0.0, 0.0]], [0, 0], [1.0, 1.0])
  → [[0.25, -0.25], [0.25, -0.25]]     same two steps, now averaged over n = 2

reinforce_grad([[0.0, 0.0]], [0], [-1.0])
  → [[-0.5, 0.5]]                      a bad return pushes the sampled action down

reinforce_grad([[0.0, 0.0]], [0], [2.0], baseline=2.0)
  → [[0.0, 0.0]]                       an exactly average outcome teaches nothing

every row of the result sums to 0.0: the gradient can only move probability between actions
```

## Constraints

- `n` at most 1 024, `A` at most 16; float64 numpy.
- numpy only: no torch, no autograd, no finite differences in your solution.
- Do not mutate `logits`, `actions` or `returns`.

## Hints

1. Write `log pi(a) = logits[a] - log sum_j exp(logits[j])`. Differentiate that with respect to `logits[j]` for `j == a` and for `j != a`. What single expression covers both cases?
2. A common wrong answer differentiates `pi(a)` instead of `log pi(a)`. The two differ by exactly one factor. What is it, and what would it do to a step whose sampled action was already very unlikely?
3. Two steps have returns `5.0` and `7.0`. Subtract a baseline of `6.0`. Did the *average* gradient over many such batches change? Did the spread between the two rows' gradients change?
4. Why must each row of the gradient sum to zero? What constraint on `p` forces it, and which bug would show up first as a row that does not?

## Explain-back

- Nothing in `reinforce_grad` mentions transitions or next states. What does the algorithm need in order to learn, and what did it give up by not having a model of the environment?
- Two batches sampled from the identical policy produce very different gradients. Is the estimator wrong, biased, or just noisy? What does the baseline do about it, and what does it not do?
- `returns[i]` is one sampled outcome, while `q` in the bandit exercise was an average. Which of the two is a reward and which is a value, and which one does REINFORCE multiply the log-probability by?
- For an LLM, one "step" is a whole sampled response and the return arrives only at the end. Which part of `grad[i, j]` corresponds to "this token", and why does a single bad final reward smear blame across every token in the response?
