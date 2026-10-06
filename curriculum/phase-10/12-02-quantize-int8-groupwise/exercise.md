# Absmax int8 and group-wise quantization

Topic: 12. Inference II: KV cache, quantization and context windows
Difficulty: 2 of 3

## Problem

Quantization stores a float weight vector as small integers plus one scale, cutting memory and bandwidth by 4× (int8) or 8× (int4). Implement symmetric **absmax** quantization in pure Python on lists.

- `quantize_absmax_int8(xs: list[float]) -> tuple[list[int], float]`. With `qmax = 127`: `scale = max(|x|) / qmax`, and each `q_i = round(x_i / scale)` clamped to `[-127, 127]`. Return `(q, scale)`. `q` holds Python `int`s. If every entry is 0, return all-zero `q` and `scale = 0.0` (no division by zero). Round to the nearest integer; an exact `.5` may go either way (the tests avoid ties). Raise `ValueError` on an empty list.
- `dequantize(q: list[int], scale: float) -> list[float]` returns `q_i × scale`.
- `quantize_groupwise(xs: list[float], group: int, bits: int) -> tuple[list[int], list[float]]` splits `xs` into consecutive groups of `group` entries (the last group may be shorter) and quantizes each group on its own with `qmax = 2^(bits − 1) − 1` (127 for 8 bits, 7 for 4 bits), the same absmax rule as above. Return the flat `q` (same length as `xs`) and one scale per group, in order. Raise `ValueError` if `xs` is empty, `group < 1`, or `bits` is outside `2 … 8`.
- `dequantize_groupwise(q: list[int], scales: list[float], group: int) -> list[float]` undoes it.

Guarantees the tests check: the reconstruction error of every element is at most `scale / 2` of its group (that is what rounding to nearest buys you), exact zeros come back as exactly `0.0`, and in the group-wise version an outlier in one group does not enlarge the error in any other group.

## Examples

```
quantize_absmax_int8([0.0, 1.0, -5.0, 2.0])   → ([0, 25, -127, 51], 0.03937...)   scale = 5 / 127
dequantize([0, 25, -127, 51], 5 / 127)          → [0.0, 0.984..., -5.0, 2.007...]
quantize_absmax_int8([0.0, 0.0])                → ([0, 0], 0.0)

quantize_groupwise([100.0, 0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.05], 4, 8)
    → q = [127, 0, 0, 0, 42, -85, 127, 21], scales = [100 / 127, 0.3 / 127]
quantize_groupwise([1.0, -0.6, 0.25], 3, 4)     → ([7, -4, 2], [1 / 7])          4-bit: qmax = 7, 0.25 / (1/7) = 1.75 → 2
dequantize_groupwise([7, -4, 2], [1 / 7], 3)    → [1.0, -0.571..., 0.285...]
```

## Constraints

- Lists have at most 10 000 entries. Pure Python with `math`.
- Absolute tolerance in the tests: 1e-9 on scales, `scale / 2 + 1e-12` on reconstruction error.

## Hints

1. What single number do you need to map the largest magnitude in the vector onto 127? What happens to that mapping when the largest magnitude is 0?
2. `int(x)` and `round(x)` differ for `x = 125.7`. Which one keeps every element within half a step of its original, and why does the error bound depend on that?
3. If one weight is 100 and the rest are around 0.1, what is `scale` and what does `round(0.1 / scale)` give? How does splitting into groups change the answer for the small weights?
4. How do you cut a list into chunks of `group` with a range step, and how do you know which scale belongs to `q[i]` when dequantizing?

## Explain-back

- Int8 weights are 4× smaller than fp32. Does that make the matmul 4× faster on its own? Which resource does quantization save first, and when does it also speed up compute?
- Why does one outlier weight ruin per-tensor absmax quantization for everything else, and how do group-wise scales, LLM.int8's outlier columns and SmoothQuant each attack that?
- Absmax is symmetric: it wastes range when the weights are all positive. What does a zero-point add, and what must stay exactly representable after quantization?
- PTQ quantizes a trained model; QAT trains with fake quantization in the loop. What does QAT let the model do that PTQ cannot, and why is 4-bit where the difference shows?
