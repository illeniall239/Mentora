# Zero-shot classification with CLIP embeddings

Topic: 2. Contrastive learning and CLIP
Difficulty: 1 of 3

## Problem

CLIP classifies without ever being trained on your classes. You embed the image once, embed one prompt per candidate label ("a photo of a dog", "a photo of a cat", ...), and pick the label whose text embedding points most nearly in the same direction as the image embedding. Write that last step.

`zero_shot_classify(img_emb: np.ndarray, label_embs: np.ndarray) -> int`

- `img_emb` is a 1-D `float64` array of shape `(d,)`: one image embedding straight out of the image encoder's projection, **not** normalized.
- `label_embs` is a 2-D array of shape `(K, d)`: one text embedding per candidate label, in label order, also not normalized.

Return the integer index in `[0, K)` of the label with the largest **cosine similarity**

```
cos(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

to `img_emb`. L2-normalize both sides first (or divide by the norms afterwards — same thing); a raw dot product is not the answer, because the encoders put no constraint on the length of an embedding and a merely long text embedding would win on magnitude alone. The consequence to rely on: multiplying `img_emb`, or any single row of `label_embs`, by any positive scalar must not change the answer.

Break ties by returning the **smallest** index. The score may be negative for every label; still return the argmax — this function always names a label, which is exactly why a zero-shot classifier cannot say "none of these".

Do not modify the inputs. Raise `ValueError` if `img_emb` is not 1-D, if `label_embs` is not 2-D, if the widths disagree, if `K == 0`, or if `img_emb` or any row of `label_embs` has zero norm (a direction that does not exist).

## Examples

```
zero_shot_classify(array([1., 0.]), array([[0., 1.], [1., 0.]]))         → 1
zero_shot_classify(array([3., 0.]), array([[0., 1.], [1., 0.]]))         → 1          scale changes nothing
zero_shot_classify(array([1., 0.]), array([[9., 1.], [1., 0.]]))         → 1          cosine 0.994 < 1.000,
                                                                                      though the raw dot is 9 > 1
zero_shot_classify(array([1., 0.]), array([[-1., 0.], [0., -1.]]))       → 1          -1.0 vs 0.0, both bad
zero_shot_classify(array([1., 1.]), array([[1., 1.], [1., 1.]]))         → 0          tie, lowest index
zero_shot_classify(array([1., 0.]), array([[0., 0.], [1., 0.]]))         → ValueError
```

## Constraints

- `K` up to 1000, `d` up to 512.
- numpy only; no loops over `K` are needed.
- Scores are compared within `1e-9`, so exact ties in the examples are exact.

## Hints

1. Which two quantities does cosine similarity divide the dot product by, and what does each one remove from the comparison?
2. `img_emb` is `(d,)` and `label_embs` is `(K, d)`. Which single matrix-vector product gives you all `K` dot products at once, and what shape comes back?
3. `np.linalg.norm` on a 2-D array needs to know which axis holds a label's features and whether to keep that axis. Which `axis` and which `keepdims` give you a `(K, 1)` column you can divide `label_embs` by?
4. Does dividing `img_emb` by its own norm ever change which label wins? If not, which of the two normalizations is doing the real work here — and would that still be true if you were comparing scores across two different images?

## Explain-back

- CLIP was never shown your label set during training. What is actually being compared here, and where did the "classifier" come from?
- What happens to the result if you skip normalization entirely? Construct a case where the wrong label wins, and say which encoder property makes that possible.
- This function always returns a label. What does that mean for an image of something that is on none of your prompts, and what would you need to add to detect that?
- The prompts are usually "a photo of a {label}" rather than just "{label}", and good pipelines average the embeddings of several templates. Why would that change the answer at all?
