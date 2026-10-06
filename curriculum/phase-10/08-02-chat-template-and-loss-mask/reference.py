# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
ROLES = ("system", "user", "assistant")


def apply_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if not messages:
        raise ValueError("no messages")
    parts = []
    for m in messages:
        if m["role"] not in ROLES:
            raise ValueError(f"unknown role {m['role']!r}")
        parts.append(f"<|{m['role']}|>\n{m['content']}<|end|>\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def loss_mask(tokens: list[str], roles: list[str]) -> list[int]:
    if len(tokens) != len(roles):
        raise ValueError("tokens and roles must align")
    if any(r not in ROLES for r in roles):
        raise ValueError("unknown role")
    return [int(role == "assistant" and tok != "<|assistant|>") for tok, role in zip(tokens, roles)]
