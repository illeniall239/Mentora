# Training FLOPs and the Chinchilla rule

Topic: 7. Scaling and pretraining
Difficulty: 1 of 3

## Problem

Two back-of-the-envelope formulas that every scaling discussion starts from. Pure Python only.

- `train_flops(n_params: int | float, n_tokens: int | float) -> float` returns the approximate compute of one training run, `6 · N · D`: for every parameter and every token, about 2 FLOPs in the forward pass and 4 in the backward pass (gradients with respect to activations and to weights). `N` counts the non-embedding parameters, `D` the training tokens. Raise `ValueError` if either argument is not positive.
- `chinchilla_tokens(n_params: int | float) -> float` returns the compute-optimal number of training tokens for `N` parameters under the Chinchilla rule of thumb, `20 · N`. Raise `ValueError` if `n_params` is not positive.

Both return floats.

## Examples

```
train_flops(1, 1)                 → 6.0
train_flops(70e9, 1.4e12)         → 5.88e23
chinchilla_tokens(70e9)           → 1.4e12
chinchilla_tokens(1e9)            → 2e10
train_flops(1e9, chinchilla_tokens(1e9))   → 1.2e20
train_flops(0, 100)               → ValueError
```

## Constraints

- Inputs are up to 1e15; results are compared with a relative tolerance of 1e-9.
- Pure Python only.

## Hints

1. A matmul of an `(n, d)` input against `(d, k)` weights takes about `2 · n · d · k` FLOPs (one multiply, one add). Per token and per weight, how many is that in the forward pass?
2. The backward pass computes two gradients per layer: one with respect to the layer's input and one with respect to its weights. How does that turn 2 into 6?
3. Chinchilla found that at fixed compute you should grow parameters and tokens at the same rate. With `D = 20 N`, how does the total compute grow if you double `N`?
4. Which of your two functions does a validation check belong in, and what happens to `6 · N · D` if `N` or `D` is zero or negative?

## Explain-back

- Gopher had 280B parameters and Chinchilla 70B, trained with about the same compute. How many tokens did each see, and why did the smaller one win?
- A forward pass at inference costs about `2 · N` FLOPs per token, not `6 · N`. What is missing, and why does that matter for a model that will serve billions of queries?
- Llama-style models are trained on far more than 20 tokens per parameter. Is that a mistake? What are they optimising instead of training compute?
- Does `train_flops` know anything about whether the model will follow instructions? What does pretraining on `D` web tokens actually teach, and what does it not?
