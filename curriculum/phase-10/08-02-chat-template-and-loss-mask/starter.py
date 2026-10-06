def apply_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render each message as '<|role|>\n{content}<|end|>\n'; optionally append '<|assistant|>\n'."""
    raise NotImplementedError


def loss_mask(tokens: list[str], roles: list[str]) -> list[int]:
    """1 for assistant-turn tokens except the opening <|assistant|> tag, 0 for everything else."""
    raise NotImplementedError
