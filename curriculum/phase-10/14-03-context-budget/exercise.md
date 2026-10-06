# Budget the context window

Topic: 14. Tokenization revisited and LLM quirks
Difficulty: 2 of 3

## Problem

A context window is measured in **tokens**, and it has to hold the whole prompt *and* the reply the model has not written yet. Budget it. Pure Python only.

Token counts come from a tokenizer; here they are handed to you as integers, one per message, so this exercise is pure arithmetic and list surgery.

`fits_context(message_token_counts: list[int], limit: int, reserve: int) -> bool` returns whether the conversation still leaves room for an answer:

```
fits  ⟺  sum(message_token_counts) + reserve <= limit
```

`reserve` is the space held back for the model's own output; it counts against the window exactly like a prompt token does. An empty list is a conversation of 0 tokens. Raise `ValueError` if `limit < 0`, `reserve < 0`, or any count is negative.

`trim_history(messages: list[str], counts: list[int], limit: int, reserve: int) -> list[str]` drops turns until it fits:

- `messages[0]` is the **system prompt** and is never dropped, whatever happens.
- `counts[i]` is the token count of `messages[i]`.
- While the kept messages do not satisfy `fits_context`, drop the **oldest surviving non-system message** — index 1, then index 2, and so on. Never drop from the newest end.
- Return a **new** list of the kept messages in their original order. Do not modify `messages` or `counts`.
- Raise `ValueError` if `messages` is empty, if `len(messages) != len(counts)`, if `limit < 0` or `reserve < 0` or any count is negative, or if the system prompt alone still does not fit (`counts[0] + reserve > limit`) — that is a broken configuration, not a history to trim.

## Examples

```
fits_context([10, 30, 40, 20, 25], 100, 20)  → False    125 + 20 = 145 > 100
fits_context([10, 20, 25], 100, 20)          → True     55 + 20 = 75
fits_context([80], 100, 20)                  → True     80 + 20 = 100, exactly full
fits_context([81], 100, 20)                  → False    one token over
fits_context([], 100, 20)                    → True
fits_context([], 10, 20)                     → False    the reserve alone does not fit
fits_context([10], 100, -1)                  → ValueError

MESSAGES = ["sys", "u1", "a1", "u2", "a2"]
COUNTS   = [   10,   30,   40,   20,   25]

trim_history(MESSAGES, COUNTS, 100, 20)  → ["sys", "u2", "a2"]
#   145 > 100 → drop "u1" → 115 > 100 → drop "a1" → 75 ≤ 100, stop
trim_history(MESSAGES, COUNTS, 200, 20)  → ["sys", "u1", "a1", "u2", "a2"]
trim_history(MESSAGES, COUNTS, 40, 5)    → ["sys", "a2"]
trim_history(MESSAGES, COUNTS, 10, 0)    → ["sys"]
trim_history(["sys"], [10], 100, 95)     → ValueError    10 + 95 > 100
```

## Constraints

- Up to 10 000 messages; counts, `limit` and `reserve` are Python `int`s.
- `fits_context` returns a `bool`; `trim_history` returns a new `list[str]` and leaves its arguments untouched.
- Pure Python only.

## Hints

1. The window has to hold the prompt and the answer. If the limit is 8 000 tokens and you pack 8 000 tokens of history into it, how long an answer can the model write?
2. Which comparison ends your loop — `<` or `<=`? Say in words what `sum + reserve == limit` means for the model, and decide which side of the line it belongs on.
3. A conversation runs system, user, assistant, user, assistant. Exactly one message must survive every trim; the rest get dropped from one particular end. Which message, and which end?
4. `counts[0] + reserve > limit` before you drop anything. Is there any sequence of drops that rescues it? What should the function do instead of looping?

## Explain-back

- An app that worked against a 4 000-token window starts failing with "context length exceeded" even though the prompt is comfortably under the limit. Which term of the budget was left out, and why does it only bite on long answers?
- The counts come from a tokenizer. Why can you not budget this in words or characters instead, and what would go wrong first for a user writing Japanese?
- You switch to a different provider's model with the *same* advertised window size, change nothing in your prompts, and the same conversation no longer fits. What changed?
- Dropping the oldest turns is the cheapest policy. What does it throw away that you may need, and — given that models degrade on material buried in the middle of a long window — what would you do instead of simply filling the window to the brim?
