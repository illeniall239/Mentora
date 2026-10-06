# The two GAN losses

Topic: 4. GANs, briefly
Difficulty: 1 of 3

## Problem

A GAN trains two networks against each other. The discriminator `D` outputs a probability that its input is real; the generator `G` wants that probability to be high for its fakes. Write the two losses, each from probabilities that `D` has already produced.

### `d_loss(d_real: np.ndarray, d_fake: np.ndarray) -> float`

The discriminator's loss — binary cross-entropy with the label 1 on real samples and 0 on fakes:

```
d_loss = -mean(log(d_real)) - mean(log(1 - d_fake))
```

`d_real` holds `D(x)` for a batch of real samples and `d_fake` holds `D(G(z))` for a batch of fakes. The two batches are averaged **separately** and may have different lengths; the result is a sum of two means, not a mean of a sum and not a sum of sums. `D` minimizes this: it falls as `d_real` moves towards 1 and `d_fake` towards 0.

### `g_loss_nonsat(d_fake: np.ndarray) -> float`

The **non-saturating** generator loss:

```
g_loss_nonsat = -mean(log(d_fake))
```

The generator maximizes `log D(G(z))` rather than minimizing `log(1 - D(G(z)))`. The two have the same fixed point but not the same gradients: when the discriminator is winning and `d_fake` is near 0 — exactly when the generator most needs a signal — the saturating form flattens out towards 0 while this one blows up towards `+inf`. If your `g_loss_nonsat` returns something *small* for `d_fake = 0.01`, you have written the saturating version.

### Both

Clamp every probability into `[1e-12, 1 - 1e-12]` before taking any logarithm, so an input of exactly 0 or 1 gives a large finite number instead of `inf` or `nan`. Return a Python `float`.

At the theoretical equilibrium, where `G` matches the data and `D` can only guess, every probability is `0.5`:

```
d_loss = 2 * ln 2 = 1.3862943611198906      g_loss_nonsat = ln 2 = 0.6931471805599453
```

Both losses sitting near those values is what "converged" looks like for a GAN — neither one marching to zero like a supervised loss.

Do not modify the inputs. Raise `ValueError` if either array is not 1-D, is empty, or holds a value outside `[0, 1]` (these are probabilities, not logits — a sigmoid has already been applied).

## Examples

```
half = full(4, 0.5)
d_loss(half, half)                      → 1.3862943611198906     2 ln 2
g_loss_nonsat(half)                     → 0.6931471805599453     ln 2

d_loss(ones(3), zeros(3))               → 1.999955756560757e-12   a perfect discriminator
g_loss_nonsat(zeros(3))                 → 27.631021115928547     and the generator screaming
g_loss_nonsat(array([0.99]))            → 0.01005033585350145

d_loss(array([0.9]), array([0.1]))      → 0.21072103131565256
d_loss(array([0.1]), array([0.9]))      → 4.605170185988092      the same numbers, swapped round

d_loss(full(2, 0.5), full(7, 0.5))      → 1.3862943611198906     means, so batch sizes may differ
d_loss(array([1.5]), array([0.1]))      → ValueError
g_loss_nonsat(array([]))                → ValueError
```

## Constraints

- Batches up to 4096 each; the two batches need not be the same length.
- numpy only.
- Compared within `1e-9`.

## Hints

1. Which label does the discriminator want for a real sample and which for a fake? Write the binary cross-entropy for each and see which one needs `1 - p`.
2. The two terms average over different batches. What goes wrong if you concatenate them and take one mean when the batches have different sizes?
3. Sketch `-log(p)` and `log(1 - p)` for `p` near 0. Which has a steep slope there, and why does the generator care about the slope rather than the value?
4. What does `log(0)` give you in numpy, and what does that single value do to a mean? Where exactly does the clamp have to sit relative to the logarithm?

## Explain-back

- Both losses sit near `ln 2` and `2 ln 2` and stop moving. Is that a converged GAN, a dead generator or a broken discriminator? What else would you look at to tell?
- Why is the non-saturating form the one everybody uses? Describe what the generator's gradient looks like early in training under each form.
- Once training is finished, which of the two networks do you keep and which do you throw away? What is the discarded one for?
- Your generator produces flawless, photorealistic samples, and `d_fake` sits near `0.5`. Name a failure this does not rule out, and say which of these two numbers would reveal it.
