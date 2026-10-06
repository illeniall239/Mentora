# Reparameterization and the Gaussian KL

Topic: 3. Autoencoders, VAEs and VQ-VAE
Difficulty: 1 of 3

## Problem

A VAE encoder does not output a latent vector. It outputs the parameters of a distribution — a mean `mu` and a log-variance `log_var` — and the decoder gets a *sample* from it. Two pieces make that trainable: the reparameterization trick, which moves the randomness into an input so gradients can flow, and the closed-form KL that pulls the posterior towards the prior. Write both with plain torch tensor ops.

### `reparameterize(mu: torch.Tensor, log_var: torch.Tensor, eps: torch.Tensor) -> torch.Tensor`

```
z = mu + exp(0.5 * log_var) * eps
```

The network outputs `log_var = log(sigma ** 2)`, not `sigma` and not `sigma ** 2` — that is why the exponent is `0.5 * log_var`: it is the **standard deviation** that multiplies `eps`, never the variance. `eps` is supplied by the caller (so these tests are deterministic) and must be drawn from `N(0, I)`, the *standard* normal — the mean and spread live entirely in `mu` and `log_var`. Shapes all match and the result has the same shape; `eps = 0` returns `mu` exactly. Gradients must reach both `mu` and `log_var`: do not detach anything, do not use `torch.no_grad`, and do not sample inside the function.

### `kl_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor`

The closed-form `KL(N(mu, diag(sigma ** 2)) || N(0, I))` for a diagonal Gaussian:

```
kl = -0.5 * sum_over_latent_dims(1 + log_var - mu ** 2 - exp(log_var))
```

`mu` and `log_var` have shape `(B, D)` where `D`, the **last** dimension, is the latent dimension. Sum over that dimension and nothing else, so the result has shape `(B,)`: one KL per example, not one number for the batch and not one per latent dimension. The caller decides whether to average over the batch.

Watch the sign and the factor. KL is a divergence: it is `>= 0` for every input, and exactly `0` when `mu = 0` and `log_var = 0` (that is, `sigma = 1`), which is the prior itself. It grows as `mu` moves away from 0 in either direction and as `log_var` moves away from 0 in either direction. If you get negative values, the sign is inside out.

Raise `ValueError` from both functions if the argument shapes do not all match, or if the tensors are not 2-D for `kl_gaussian`.

## Examples

```
reparameterize(tensor([1., 2.]), tensor([0., 1.3862944]), tensor([1., -1.]))
                                                  → [2., 0.]        sigma = [1, 2], not [1, 4]
reparameterize(mu, log_var, zeros_like(mu))       → mu
reparameterize(mu, zeros_like(mu), eps)           → mu + eps

kl_gaussian(zeros(1, 4), zeros(1, 4))             → [0.]
kl_gaussian(tensor([[1., 0.]]), zeros(1, 2))      → [0.5]
kl_gaussian(zeros(1, 1), tensor([[1.]]))          → [0.3591409]     -0.5 * (1 + 1 - 0 - e)
kl_gaussian(zeros(1, 1), tensor([[-1.]]))         → [0.1839397]     a too-narrow posterior also costs
kl_gaussian(randn(8, 5), randn(8, 5))             → shape (8,), every entry >= 0
```

## Constraints

- `B` up to 1024, `D` up to 64. `float32` or `float64`, whatever comes in.
- torch tensor ops only. No `torch.distributions`, no `F.kl_div`, no sampling inside either function.
- Compared against `torch.distributions.kl_divergence` within `1e-5`.

## Hints

1. The encoder emits `log_var`. Write down, in one line each, how you get `sigma ** 2` and `sigma` from it. Which one belongs next to `eps`?
2. Gradients flow through `mu` and `log_var` but not through `eps`. Which operations in `z = mu + sigma * eps` are differentiable with respect to what, and why would sampling *inside* the function break backpropagation?
3. In the KL formula, which dimension do you sum over, and what shape does that leave? What would `.sum()` with no argument give you instead, and when would you notice the difference?
4. Put `mu = 0, log_var = 0` into the formula term by term. What has to be true of the four terms for the total to come out exactly zero — and what does that tell you about the sign in front of the `0.5`?

## Explain-back

- Why can you not just sample `z ~ N(mu, sigma ** 2)` inside the forward pass and call `.backward()`? What exactly does the reparameterization move out of the way?
- `eps` must come from `N(0, I)`. What happens to the latents the decoder sees if you draw it from `N(0, 4I)` instead, or from a uniform distribution?
- A plain autoencoder has an encoder, a decoder and a bottleneck, and can reconstruct beautifully — yet you cannot sample new data from it. Which term here is what makes a VAE generative, and what does it do to the latent space?
- If you deleted the KL term and kept only reconstruction, what would `log_var` drift towards during training, and what would the latent space look like when you then tried to sample from `N(0, I)`?
