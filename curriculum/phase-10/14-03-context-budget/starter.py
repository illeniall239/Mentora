def fits_context(message_token_counts: list[int], limit: int, reserve: int) -> bool:
    """Whether sum(message_token_counts) + reserve fits within limit."""
    raise NotImplementedError


def trim_history(messages: list[str], counts: list[int], limit: int, reserve: int) -> list[str]:
    """Keep messages[0] and drop the oldest other messages until the conversation fits."""
    raise NotImplementedError
