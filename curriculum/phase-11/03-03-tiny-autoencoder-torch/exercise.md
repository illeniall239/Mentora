# A tiny autoencoder in PyTorch

Topic: 3. Autoencoders, VAEs and VQ-VAE
Difficulty: 3 of 3

## Problem

Build the smallest thing that deserves the name autoencoder — encoder, bottleneck, decoder — and train it until it reconstructs its data almost exactly. The data lies on a low-dimensional linear manifold inside a wider space, so a narrow enough bottleneck loses nothing and a too-narrow one cannot be rescued by training longer.

### `AE(d_in: int, d_latent: int)`

An `nn.Module` with exactly two layers and no activations:

- `self.encoder = nn.Linear(d_in, d_latent)`
- `self.decoder = nn.Linear(d_latent, d_in)`

and three methods:

- `encode(x)` — `(N, d_in)` to `(N, d_latent)`, just the encoder.
- `decode(z)` — `(N, d_latent)` to `(N, d_in)`, just the decoder.
- `forward(x)` — `decode(encode(x))`, so `forward` and the composition agree exactly.

Both layers keep their default biases, so `AE(8, 3)` holds `8*3 + 3 + 3*8 + 8 = 59` parameters. Raise `ValueError` from `__init__` if `d_in < 1` or `d_latent < 1`.

### `train_ae(data, d_latent, steps=600, lr=0.05, seed=0) -> tuple[AE, float]`

`train_ae(data: torch.Tensor, d_latent: int, steps: int = 600, lr: float = 0.05, seed: int = 0)`

1. Call `torch.manual_seed(seed)` **before** constructing the model, so the run is reproducible: two calls with the same arguments return the same final loss, bit for bit.
2. Build `AE(data.shape[1], d_latent)`.
3. Optimize with `torch.optim.Adam(model.parameters(), lr=lr)` for `steps` full-batch steps. Each step: zero the gradients, compute the mean squared error between `model(data)` and `data` (`F.mse_loss`, mean reduction over every element), backpropagate, step.
4. Return `(model, final_loss)` where `final_loss` is a Python `float`: the MSE of the trained model on `data`, measured after the last step under `torch.no_grad()`.

Raise `ValueError` if `data` is not a 2-D tensor or if `steps < 1`.

With the defaults, on 256 points that lie exactly on a 3-dimensional linear manifold in 8-dimensional space, `d_latent = 3` drives the loss below `1e-6` (in practice around `1e-13`) while `d_latent = 1` stalls above `0.5` however long you run it. The trained model reconstructs *unseen points from the same manifold* just as well, and reconstructs points off the manifold badly — it has learned a subspace, not a copy of the training set.

What it cannot do: generate. There is no prior over `z` here, so `decode(torch.randn(n, d_latent))` is not a sample from the data distribution; it is whatever the decoder happens to map a random point to. That gap is exactly what the KL term of a VAE closes.

## Examples

```
m = AE(8, 3)
m.encode(torch.randn(5, 8)).shape             → torch.Size([5, 3])
m.decode(torch.randn(5, 3)).shape             → torch.Size([5, 8])
sum(p.numel() for p in m.parameters())        → 59
AE(8, 0)                                      → ValueError

model, loss = train_ae(manifold_data, 3)      → loss < 1e-6
model, loss = train_ae(manifold_data, 1)      → loss > 0.5
train_ae(manifold_data, 3)[1] == train_ae(manifold_data, 3)[1]   → True
train_ae(torch.randn(8), 3)                   → ValueError
```

## Constraints

- CPU only, `float32`, at most 1024 points and `d_in <= 32`. The whole test file runs in a couple of seconds.
- `torch` and `torch.nn` only; no `nn.Sequential` requirement either way, but the two attribute names `encoder` and `decoder` are part of the interface.
- Full batch: no `DataLoader`, no minibatching, no shuffling.

## Hints

1. Which `nn.Module` boilerplate has to run before you can assign submodules in `__init__`, and what breaks if you forget it?
2. `forward` must equal `decode(encode(x))`. If you write the arithmetic out a second time inside `forward` instead of calling the two methods, what can drift apart?
3. The three lines of an optimization step always come in the same order. Which one has to happen *before* `backward`, and what accumulates if it does not?
4. After the loop, `model(data)` still builds a graph. Which context manager stops that, and why does it matter for the number you return rather than for correctness?

## Explain-back

- With `d_latent = 3` the loss reaches roughly `1e-13`; with `d_latent = 1` it stalls. What is the bottleneck actually doing, and what does the number 3 have to do with the data?
- This model reconstructs held-out points from the same manifold perfectly. Is it storing the training set? What is it storing?
- You now have a decoder that maps vectors to plausible-looking data. Why is this still not a generative model, and what would you have to add to sample from it?
- A linear autoencoder with MSE recovers the same subspace as PCA. What does adding a nonlinearity between the layers buy, and what does it cost you in terms of being able to say what the latent space means?
