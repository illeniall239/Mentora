# The KL-penalized reward

Topic: 10. RLHF and preference optimization (DPO)
Difficulty: 2 of 3

## Problem

In the PPO stage of RLHF the thing being maximised is not the reward model's score. It is the score **minus** a penalty for drifting away from the frozen reference model the policy started as. Without that penalty the policy walks off into whatever gibberish the reward model happens to score highly. Implement the per-sequence objective and the batch KL estimate.

A batch of `B` sampled responses, each `T` tokens long, arrives as per-token log-probabilities of the tokens the policy actually sampled:

- `logp_policy` — float tensor of shape `(B, T)`, `log pi(token_t | prefix)` under the current policy.
- `logp_ref` — float tensor of shape `(B, T)`, the same tokens scored by the frozen reference model.
- `r` — float tensor of shape `(B,)`, the reward model's score for the whole response. It arrives once, at the end.
- `beta` — a non-negative float.

**`kl_penalized_reward(r, logp_policy, logp_ref, beta) -> torch.Tensor`** returns a tensor of shape `(B,)`:

```
kl_b  = sum over t of (logp_policy[b, t] - logp_ref[b, t])      summed over tokens, not averaged
out_b = r[b] - beta * kl_b
```

The log-ratio is `policy minus reference` and the penalty is **subtracted**. Drifting from the reference must lower the objective, never raise it.

**`kl_estimate(logp_policy, logp_ref) -> torch.Tensor`** returns a 0-dimensional tensor: the same per-sequence sum, averaged over the batch.

```
kl_estimate = mean over b of (sum over t of (logp_policy[b, t] - logp_ref[b, t]))
```

- This is the single-sample (k1) estimator of `KL(pi || pi_ref)`. It is unbiased, but on any one batch it can come out **negative** — a sampled response the reference happened to like better than the policy does. Do not clamp it, do not take an absolute value: the tests check a negative case exactly.
- Both functions raise `ValueError` if `logp_policy` and `logp_ref` are not both 2-D with the same shape, if `B == 0` or `T == 0`, and `kl_penalized_reward` additionally if `r` is not 1-D of shape `(B,)` or if `beta < 0`.
- Both results stay differentiable: the test backpropagates through them. Do not call `.detach()`, `.item()`, `torch.no_grad()` or convert to numpy anywhere in your solution.
- Tolerance for all comparisons is `1e-5`.

## Examples

```
logp_policy = [[-1.0, -2.0, -0.5]];  logp_ref = [[-1.0, -2.0, -0.5]];  r = [4.0]
kl_estimate(logp_policy, logp_ref)                      → 0.0
kl_penalized_reward(r, logp_policy, logp_ref, 0.2)      → [4.0]        no drift, no penalty

logp_policy = [[-1.0, -1.0, -1.0, -1.0]];  logp_ref = [[-1.5, -1.5, -1.5, -1.5]];  r = [4.0]
kl_estimate(...)                                        → 2.0          0.5 per token, 4 tokens
kl_penalized_reward(r, ..., beta=0.2)                   → [3.6]        4.0 - 0.2 * 2.0
kl_penalized_reward(r, ..., beta=1.0)                   → [2.0]
kl_penalized_reward(r, ..., beta=0.0)                   → [4.0]        the unpenalized reward

two sequences whose per-sequence KLs are 2.0 and 0.0:
kl_estimate(...)                                        → 1.0          mean over the batch

logp_policy = [[-3.0]];  logp_ref = [[-1.0]]
kl_estimate(...)                                        → -2.0         negative on one sample, and that is fine
```

## Constraints

- `B` at most 32, `T` at most 64; float32 CPU tensors; PyTorch only, no numpy.
- No `torch.nn.functional.kl_div` and no `torch.distributions`: these are sampled log-probs of chosen tokens, not full distributions, so write the sum yourself.
- Do not modify the inputs in place.

## Hints

1. `kl_b` has one number per sequence, but the inputs have one number per token. Which dimension do you reduce, and with which reduction? What would averaging over `T` instead mean for a 10-token answer versus a 1 000-token one?
2. Write the objective for a policy that has drifted a long way and earned a big reward. Which of the two terms must win as `beta` grows, and what does that tell you about the sign in front of the penalty?
3. The estimator is `log pi - log pi_ref` for tokens *sampled from pi*. Why is the expectation of that non-negative, while any single sequence can give a negative number?
4. If you called `.item()` on the penalty before subtracting it, the forward value would still be right. What would break at the backward pass, and what would the policy then be trained to do?

## Explain-back

- Someone proposes dropping the KL term to "let the model optimise the reward properly". Describe what their policy produces after a few thousand steps, and why the reward model will happily score it highly.
- The KL estimate came out negative on one batch. Is the code wrong, or is something else going on? What would you look at over 100 batches before worrying?
- `r` is one number for an entire response, while the penalty is a sum over every token. What does that asymmetry say about which part of RLHF carries the fine-grained signal?
- Both the reward model and the reference model are frozen copies of other models. What would go wrong if the reference were updated to the current policy at every step?
