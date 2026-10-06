# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def fits_context(message_token_counts: list[int], limit: int, reserve: int) -> bool:
    if limit < 0 or reserve < 0 or any(c < 0 for c in message_token_counts):
        raise ValueError("limit, reserve and every token count must be non-negative")
    return sum(message_token_counts) + reserve <= limit


def trim_history(messages: list[str], counts: list[int], limit: int, reserve: int) -> list[str]:
    if not messages:
        raise ValueError("messages must contain at least the system prompt")
    if len(messages) != len(counts):
        raise ValueError("messages and counts must have the same length")
    fits_context(counts, limit, reserve)  # validates limit, reserve and every count
    if not fits_context(counts[:1], limit, reserve):
        raise ValueError("the system prompt alone does not fit in the limit")
    first = 1
    while not fits_context([counts[0]] + counts[first:], limit, reserve):
        first += 1
    return [messages[0]] + messages[first:]
