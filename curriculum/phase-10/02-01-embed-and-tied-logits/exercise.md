# Embed and tied logits

Topic: 2. Embeddings revisited: static, contextual and tied
Difficulty: 1 of 3

## Problem

A token embedding is a lookup into a learned matrix, and that lookup is exactly a one-hot matrix multiply. The same matrix can also serve as the output layer ("tied" weights). Write both in PyTorch with plain tensor ops: no `torch.nn.Embedding`, no `torch.nn.functional.embedding`, no `torch.nn.Linear`.

- `embed(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor`. `ids` is an integer tensor of any shape (usually `(T,)` or `(B, T)`), `table` is `(V, D)`. Return a tensor of shape `ids.shape + (D,)` whose entry at `[..., :]` is row `ids[...]` of `table`. It must equal `one_hot(ids, V).float() @ table` within `1e-6`, but you may implement it with indexing. It must be differentiable with respect to `table`: a row looked up twice receives twice the gradient. An id outside `0 … V-1` raises `IndexError`.
- `tied_logits(h: torch.Tensor, table: torch.Tensor) -> torch.Tensor`. `h` has shape `(..., D)`; return `h @ table.T`, shape `(..., V)`: the logit for token `v` is the dot product of `h` with `v`'s embedding row. Raise `ValueError` if the last dimension of `h` is not `D`.

## Examples

```
table = [[1., 0.], [0., 1.], [2., 3.]]                       V = 3, D = 2
embed(tensor([2, 0]), table)         → [[2., 3.], [1., 0.]]
embed(tensor([[1], [1]]), table)     → shape (2, 1, 2)
tied_logits(tensor([[2., 3.]]), table)   → [[2., 3., 13.]]
tied_logits(tensor([[1., 2., 3.]]), table)  → ValueError
embed(tensor([3]), table)            → IndexError
```

## Constraints

- `V` at most 1 000, `D` at most 64, `ids` has at most 1 000 entries. float32.
- Match the one-hot formulation within `1e-6`.
- PyTorch on CPU, tensor ops only.

## Hints

1. What does `one_hot(ids) @ table` compute for one id, entry by entry? Which single row of `table` survives the sum, and which tensor operation gives you that row directly?
2. If `ids` is `(B, T)` and `table` is `(V, D)`, what shape does `table[ids]` have, and why is that the right one?
3. When the same id appears twice in `ids`, how many paths does the gradient of the loss have back to that row of `table`? What must the gradient of that row therefore be, compared with an id used once?
4. For `tied_logits`, what does `h @ table.T` measure between `h` and each vocabulary row? What is `h` in a GPT at the point where this is applied?

## Explain-back

- Is dimension 7 of an embedding "the royalty dimension"? What can you say about a single coordinate of a learned embedding, and what can you not say?
- In a GPT, is the vector for the token "bank" the same at layer 0 and at layer 12? Which of the two is the static embedding, and which is contextual?
- What does tying the input and output tables save, and what does it force to be true about the geometry of the last hidden state?
- Is an LLM's token embedding table the same thing as the sentence embeddings a retrieval system stores? What is embedded in each case?
