# Residual vector quantization

Topic: 10. Audio tokens and neural codecs
Difficulty: 2 of 3

## Problem

One codebook cannot describe a latent vector precisely: snapping to the nearest of 1024 entries throws away everything in between. A neural codec fixes that with **residual** vector quantization: codebook 1 quantizes the vector, codebook 2 quantizes what codebook 1 missed, codebook 3 quantizes what is still missing, and so on. Each stage emits one index, so one frame becomes one token per codebook.

Plain Python lists, `math` allowed, no numpy.

- `rvq_encode(vec: list[float], codebooks: list[list[list[float]]]) -> list[int]` — `codebooks[k]` is stage `k`'s codebook, a list of entries, each the same length as `vec`. Start with `residual = vec`. For each stage `k` in order: pick the index `i` of the entry of `codebooks[k]` with the smallest **squared Euclidean distance** to the *current residual* (not to `vec`), record `i`, then set `residual = residual − codebooks[k][i]`. Ties go to the smallest index. Return the list of indices, one per codebook.
- `rvq_decode(indices: list[int], codebooks: list[list[list[float]]]) -> list[float]` — the element-wise sum `Σ_k codebooks[k][indices[k]]`. Stage `k`'s index indexes stage `k`'s codebook: swapping the codebook order gives a different vector.
- `rvq_errors(vec: list[float], codebooks: list[list[list[float]]]) -> list[float]` — a list of `len(codebooks) + 1` Euclidean (L2) **norms**, not squared. Entry `0` is `‖vec‖`, the error if you send no tokens at all; entry `k` is `‖vec − rvq_decode(indices[:k], codebooks[:k])‖`, the error after the first `k` stages. The last entry is the error of the full code.

All three raise `ValueError` if: `codebooks` is empty, any codebook is empty, any entry has a different length from the others (or from `vec`), `vec` is empty, `len(indices) != len(codebooks)`, or an index is out of range for its codebook.

Do not modify the inputs.

## Examples

```
vec  = [0.9, 0.2]
cb1  = [[1.0, 0.0], [0.0, 1.0]]
cb2  = [[-0.1, 0.2], [0.9, 0.2]]

rvq_encode(vec, [cb1, cb2])   → [0, 0]
      stage 1: dist to [1,0] = 0.05, to [0,1] = 1.45  → index 0, residual = [-0.1, 0.2]
      stage 2: dist to [-0.1,0.2] = 0.0, to [0.9,0.2] = 1.0 → index 0
      (quantizing vec again instead of the residual would pick index 1 at stage 2)

rvq_decode([0, 0], [cb1, cb2])  → [0.9, 0.2]
rvq_errors(vec, [cb1, cb2])     → [0.92195..., 0.22360..., 0.0]

rvq_encode([1.0, 0.0], [[[2.0, 0.0], [1.0, 0.0]]])   → [1]
      nearest by distance is [1,0]; the largest dot product would wrongly say [2,0]

rvq_decode([0], [])             → ValueError
```

## Constraints

- Up to 8 codebooks, 64 entries each, vectors up to length 32.
- Plain Python lists in and out. Absolute tolerance 1e-9.
- Time: O(K · M · D) for `K` codebooks of `M` entries over `D` dimensions.

## Hints

1. Write stage 1 on its own: which entry of `codebooks[0]` is nearest, and what is left over once you subtract it? Give that leftover a name — it is the only thing stage 2 is allowed to see.
2. Squared distance and the dot product rank entries differently. For the residual `[1, 0]` and entries `[2, 0]` and `[1, 0]`, which does each rule choose, and which one makes the leftover smaller?
3. Do you need the square root inside the arg-min? What would taking it change about which index wins, and about the cost?
4. `rvq_errors` wants the error after 0, 1, …, K stages. Is there a quantity you are already carrying through the encode loop whose norm is exactly that?

## Explain-back

- Why does stage 2 quantize the residual rather than the original vector? What would the token streams look like if every stage quantized the original, and how much would stage 2 add?
- The errors you return shrink stage by stage. What is the bitrate cost of that shrinkage, and what happens to the sequence an LM must generate?
- A decoder receives the 8 streams out of order and sums them anyway. The sum of a set is order-independent — so why is the audio ruined?
- Codebook 1 of a trained codec carries coarse structure and codebook 8 fine detail. How does that relate to the split between semantic tokens and acoustic tokens in a speech LM?
