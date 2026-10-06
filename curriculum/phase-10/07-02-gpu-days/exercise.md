# GPU days

Topic: 7. Scaling and pretraining
Difficulty: 1 of 3

## Problem

Turn a FLOP budget into a wall-clock estimate for a training run. Pure Python only.

`gpu_days(flops: float, peak_flops: float, mfu: float, n_gpus: int) -> float` returns the number of **wall-clock days** a run of `flops` total FLOPs takes on `n_gpus` GPUs that each deliver `peak_flops` FLOPs per second at their theoretical peak, when the training code reaches a **model FLOPs utilisation** of `mfu` (a fraction in `(0, 1]`, e.g. `0.4` for 40 %):

```
gpu_days = flops / (peak_flops · mfu · n_gpus · 86 400)
```

Raise `ValueError` if `flops`, `peak_flops` or `n_gpus` is not positive, or if `mfu` is not in `(0, 1]`.

## Examples

```
gpu_days(86_400, 1, 1.0, 1)            → 1.0       one FLOP per second for one day
gpu_days(5.88e23, 1e15, 0.4, 1000)     → 17.01…    Chinchilla on 1 000 GPUs of 1 PFLOP/s at 40 % MFU
gpu_days(5.88e23, 1e15, 0.4, 2000)     → 8.51…     twice the GPUs, half the time
gpu_days(5.88e23, 1e15, 0.2, 1000)     → 34.03…    half the utilisation, twice the time
gpu_days(1e20, 1e15, 1.5, 8)           → ValueError
```

## Constraints

- Results are compared with a relative tolerance of 1e-9.
- Pure Python only.

## Hints

1. How many FLOPs does one GPU actually complete in one second if its peak is `peak_flops` but only `mfu` of that turns into useful model FLOPs?
2. With `n_gpus` GPUs working in parallel, what is the fleet's useful FLOPs per second, assuming the parallelism is perfect?
3. How many seconds are in a day, and where in the formula does that constant go so the answer comes out in days rather than seconds?
4. Which values of `mfu` are physically impossible, and which of the other inputs would make the division blow up?

## Explain-back

- Doubling `n_gpus` halved the days in your formula. In a real cluster, what stops that from being exactly true, and which kinds of parallelism (data, tensor, pipeline) pay which price?
- MFU of 40 % means 60 % of the peak is wasted. Name two things the GPU spends that time on.
- Chinchilla's `5.88e23` FLOPs came from `6 · N · D`. If you now serve that model, is `gpu_days` the number you would use to price inference? What replaces `6 · N · D`?
- Reducing precision to bf16 roughly doubles `peak_flops` on modern hardware. Does that change how many tokens the model needs to see, or only how fast it sees them?
