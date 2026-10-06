# The DPO loss

Topic: 10. RLHF and preference optimization (DPO)
Difficulty: 3 of 3

## Problem

Direct Preference Optimization throws away the reward model and the sampling loop of RLHF, and trains the policy on preference pairs directly. The trick: the Bradley-Terry loss of the *implicit* reward `beta * log(pi(y|x) / pi_ref(y|x))`. What is left is a classification loss on log-probability ratios against a frozen reference model.

Each preference pair gives four numbers: the log-probability of the chosen response and of the rejected response, under the policy being trained and under the frozen reference.

`dpo_loss(pi_c, pi_r, ref_c, ref_r, beta) -> torch.Tensor` takes four 1-D float tensors of shape `(B,)` — already summed over the tokens of each response — plus a positive float `beta`, and returns a 0-dimensional tensor:

```
logits_i = beta * ((pi_c[i] - ref_c[i]) - (pi_r[i] - ref_r[i]))
loss     = -mean_i log sigma(logits_i)
```

- `beta` multiplies **inside** the `log sigma`, not outside. Doubling `beta` does not double the loss; it sharpens the implicit reward.
- Both reference terms are part of the formula. Dropping them would still train something, but not DPO: the reference is what stops the policy from simply inflating the probability of everything it has ever been shown.
- Reduce with the **mean** over the batch.
- When the policy equals the reference (`pi_c == ref_c` and `pi_r == ref_r`) the loss is exactly `ln 2 ≈ 0.6931`, for every `beta`. That is the correct value at step 0, not 0.
- Use `torch.nn.functional.logsigmoid` (or `-softplus(-x)`): the test feeds a logit of `-200.0` and expects a finite loss near `200.0`, where `log(sigmoid(x))` would give `-inf`.
- The result stays differentiable with respect to `pi_c` and `pi_r`; the test backpropagates through it. No `.detach()`, `.item()`, `torch.no_grad()` or numpy inside your solution.
- Raise `ValueError` if the four tensors are not all 1-D with the same shape, if the batch is empty, or if `beta <= 0`.
- No `binary_cross_entropy_with_logits`, no `margin_ranking_loss`, no `torch.nn.*` loss modules. Tolerance for all comparisons is `1e-5`.

## Examples

```
pi_c = [-1.0];  pi_r = [-2.0];  ref_c = [-1.0];  ref_r = [-2.0]     policy == reference
dpo_loss(pi_c, pi_r, ref_c, ref_r, 0.1)  → 0.6931472   = ln 2
dpo_loss(pi_c, pi_r, ref_c, ref_r, 5.0)  → 0.6931472   same, for any beta

pi_c = [-1.0];  pi_r = [-2.0];  ref_c = [-1.0];  ref_r = [-1.0]     logit = beta * 1.0
dpo_loss(..., beta=0.5) → 0.4740770
dpo_loss(..., beta=1.0) → 0.3132617      not half of the beta = 0.5 value
dpo_loss(..., beta=2.0) → 0.1269280

pi_c = [-1.0];  pi_r = [-2.0];  ref_c = [-0.5];  ref_r = [-2.0];  beta = 0.5
  → 0.8259394      the reference pushed the logit to -0.25: ignoring it would give 0.4740770

pi_c = [-100.0];  pi_r = [0.0];  ref_c = [0.0];  ref_r = [0.0];  beta = 2.0
dpo_loss(...) → 200.0            finite, not inf
```

## Constraints

- `B` at most 64; float32 CPU tensors; PyTorch only, no numpy.
- Inputs are per-sequence log-probabilities (summed over tokens) — no token dimension to reduce here.
- Do not modify the inputs in place.

## Hints

1. Write the implicit reward of one response as `beta * (log pi - log pi_ref)`. Substitute the chosen and the rejected one into `-log sigma(r_chosen - r_rejected)`. What is left, and where did `beta` end up?
2. Set `pi_c = ref_c` and `pi_r = ref_r` and evaluate by hand. What is `sigma(0)`, and why is `ln 2` the answer no matter what `beta` is?
3. There are two ways to lower this loss: raise the chosen log-probability, or lower the rejected one. Differentiate the loss with respect to each and compare the magnitudes. Does the loss care which of the two the policy does?
4. Suppose you delete `ref_c` and `ref_r` from the formula. What now stops the policy from driving every log-probability in the batch toward `0`, and what would that do to text it was never shown?

## Explain-back

- DPO never samples from the policy during training, and never trains a reward model. Where did those two stages go — what plays their part in this one expression?
- Your loss reads `ln 2` on the first batch and someone reports it as a bug. Explain what value they expected, and why `ln 2` is the correct starting point.
- The reference model is frozen and is also the model you started from. Which term of the loss would go to zero if you refreshed the reference to the current policy each step, and what would training then optimise?
- RLHF with PPO keeps a separate explicit KL penalty; DPO has no such term, yet it still stays close to the reference. Where is the constraint hiding?
