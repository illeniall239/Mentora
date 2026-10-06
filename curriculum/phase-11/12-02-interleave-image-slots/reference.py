# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def interleave(
    text_ids: list[int], image_slots: list[int], image_token_id: int, n_per_image: int
) -> tuple[list[int], list[tuple[int, int]]]:
    if n_per_image < 1:
        raise ValueError("every image needs at least one placeholder token")
    if any(b < a for a, b in zip(image_slots, image_slots[1:])):
        raise ValueError("image_slots must be non-decreasing")
    if any(not 0 <= s <= len(text_ids) for s in image_slots):
        raise ValueError("every slot must be an index into text_ids (or len(text_ids))")
    if image_token_id in text_ids:
        raise ValueError("image_token_id is reserved and must not occur in text_ids")

    seq: list[int] = []
    spans: list[tuple[int, int]] = []
    k = 0
    for i in range(len(text_ids) + 1):
        while k < len(image_slots) and image_slots[k] == i:
            start = len(seq)
            seq.extend([image_token_id] * n_per_image)
            spans.append((start, len(seq)))
            k += 1
        if i < len(text_ids):
            seq.append(text_ids[i])
    return seq, spans
