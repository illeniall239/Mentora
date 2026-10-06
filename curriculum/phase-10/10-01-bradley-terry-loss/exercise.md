# The Bradley-Terry reward-model loss

Topic: 10. RLHF and preference optimization (DPO)
Difficulty: 1 of 3

## Problem

A reward model never sees "how good is this answer, out of 10". It sees pairs: a prompt, a **chosen** answer and a **rejected** answer, picked by a human. The Bradley-Terry model turns that into a loss by saying the probability a human prefers the chosen answer is `sigma(r_chosen - r_rejected)`, and training maximises the log-likelihood of the preferences that were actually recorded.

`bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor` returns the mean negative log-likelihood over the batch, as a 0-dimensional tensor:

```
loss = -mean_i log sigma(r_chosen[i] - r_rejected[i])
```

- `r_chosen` and `r_rejected` are 1-D float tensors of the same shape `(n,)`, the scalar reward the model assigns to each answer in the pair. Raise `ValueError` if either is not 1-D, if their shapes differ, or if `n == 0`.
- Reduce with the **mean** over the batch, not the sum.
- The sign is the one above: a *larger* chosen reward must give a *smaller* loss. Equal rewards give exactly `ln 2 ≈ 0.6931`, because the model is then saying the two answers are a coin flip.
- Be numerically stable. `torch.log(torch.sigmoid(x))` underflows to `-inf` around `x = -100` in float32; `torch.nn.functional.logsigmoid` (equivalently `-softplus(-x)`) does not. The test feeds a margin of `-200.0` and expects a finite loss near `200.0`.
- The result must stay in the autograd graph: the test calls `.backward()` through it and checks the gradients.
- No `binary_cross_entropy_with_logits`, no `margin_ranking_loss`, no `torch.nn.*` loss modules: write the formula. Tolerance for all comparisons is `1e-6`.

Two properties the tests pin down. The loss depends only on the **difference** of the rewards, so adding any constant to both leaves it unchanged — a reward model is only ever identified up to a shift, which is why the PPO stage normalises rewards. And swapping the two arguments satisfies exactly `loss(r_rejected, r_chosen) = loss(r_chosen, r_rejected) + mean(r_chosen - r_rejected)`.

## Examples

```
bradley_terry_loss(tensor([0.0]), tensor([0.0]))          → 0.6931472   = ln 2
bradley_terry_loss(tensor([1.0]), tensor([0.0]))          → 0.3132617   margin +1
bradley_terry_loss(tensor([0.0]), tensor([1.0]))          → 1.3132617   margin -1, and = 0.3132617 + 1
bradley_terry_loss(tensor([5.0]), tensor([0.0]))          → 0.0067153   confident and right
bradley_terry_loss(tensor([3.0, 0.0]), tensor([2.0, 1.0])) → 0.8132617  mean of 0.3132617 and 1.3132617

bradley_terry_loss(tensor([-100.0]), tensor([100.0]))     → 200.0       finite, not inf

r_c = tensor([1.0], requires_grad=True); bradley_terry_loss(r_c, tensor([0.0])).backward()
r_c.grad → [-0.2689414]     = -sigma(r_rejected - r_chosen) / n
```

## Constraints

- Batch size at most 256; float32 CPU tensors; PyTorch only, no numpy.
- No loss helper from `torch.nn` or `torch.nn.functional` other than `logsigmoid` / `softplus`.
- Do not modify the inputs in place.

## Hints

1. Write `-log sigma(x)` as a function of `x` without a `sigmoid` in it. (Start from `sigma(x) = 1 / (1 + e^-x)`.) What standard function have you just written?
2. Set `r_chosen = r_rejected` and evaluate your formula by hand. What number must come out, and what does that number mean about the preference the model is predicting?
3. Only one quantity derived from the two tensors ever enters the loss. Which one? What does that imply about a reward model that scores every answer 1 000 points higher than another one does?
4. Differentiate the loss with respect to `r_chosen` for a single pair. When the model is already very confident and correct, what happens to the size of that gradient, and what does training do with such pairs?

## Explain-back

- The humans who built this dataset never wrote down a number. Where did the scalar rewards come from, and in what sense is the trained reward model the humans' preferences rather than the humans themselves?
- Your loss is `ln 2` at the start of training, when every reward is roughly equal. Why is that the right starting point rather than, say, 0?
- If you add 1 000 to every reward this model produces, the loss does not move. Which downstream stage would notice that shift, and which would not?
- A reward model scores "lists three citations" highly because annotators liked such answers. What happens to the policy optimised against it for long enough, and what is that called?
