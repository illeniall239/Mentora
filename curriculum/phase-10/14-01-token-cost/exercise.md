# What a prompt actually costs

Topic: 14. Tokenization revisited and LLM quirks
Difficulty: 1 of 3

## Problem

A prompt is billed, and fits a context window, in **tokens** — not characters, not words. Count them. Pure Python only.

Write `token_cost(text: str, merges: dict[tuple[int, int], int]) -> int`, the number of tokens `text` becomes under this byte-level scheme:

1. Start from the UTF-8 bytes of `text`, each byte its own ID in 0–255. A character is **not** a byte: `"é"` is 2 bytes, `"🌍"` is 4.
2. Walk `merges` in the dict's own order (its insertion order, which is the order the merges were learned — `dict` preserves it). For each `(a, b) -> new_id`, make **one** left-to-right pass over the current sequence replacing every non-overlapping occurrence of the adjacent pair `a, b` with `new_id`. A merge created by an earlier entry can therefore be an input to a later one.
3. Return the length of the resulting ID list as an `int`.

Never modify `merges`. `token_cost(text, {})` is simply the number of UTF-8 bytes.

## Examples

```
MERGES = {(116, 104): 256, (256, 101): 257, (32, 257): 258, (97, 116): 259, (111, 110): 260}
#              "th"             "the"          " the"            "at"            "on"

token_cost("the", MERGES)                     → 1
token_cost("the", {})                         → 3
token_cost("hello", MERGES)                   → 5     no merge applies
token_cost("the the", MERGES)                 → 2     ids [257, 258]
token_cost("", MERGES)                        → 0

token_cost("the cat sat on the mat", MERGES)  → 13    22 characters, 13 tokens
token_cost("кот сидел на коврике", MERGES)    → 37    20 characters, 37 tokens
token_cost("héllo", MERGES)                   → 6     5 characters, 6 UTF-8 bytes
token_cost("🌍", MERGES)                      → 4     1 character, 4 UTF-8 bytes
```

## Constraints

- `text` is at most 5 000 characters and may contain any Unicode.
- `merges` has at most 512 entries; every ID in it is a non-negative `int`, and the pairs are given in learned order.
- The return value is an `int`, not a list.
- Pure Python only — no tokenizer library, nothing downloaded.

## Hints

1. The function is handed a `str`, but the scheme starts from something else. What is `"héllo"` before any merge is applied, and how many items long is it?
2. One merge turns two adjacent IDs into one. If you make a single pass per merge entry, how does the sequence length change after each entry, and when does it stop changing?
3. Entry `(256, 101) -> 257` mentions ID 256, which only exists after an earlier entry ran. What does walking the dict in its stored order guarantee about that?
4. The Russian line has fewer characters than its English translation but costs nearly three times as many tokens. Which bytes does Cyrillic text use, and can any of them appear in a merge table learned from English text?

## Explain-back

- Someone tells you their prompt is "about 1 000 characters, so about 1 000 tokens". What do you need to know before you can turn characters into tokens at all, and when is that estimate badly wrong?
- The identical sentence is sent to two different models and billed at two different token counts. The text did not change — what did?
- You must cut 20% of a prompt's tokens. Why is deleting 20% of its characters the wrong instrument, and what would you look for instead?
- Why does a user writing Hindi or Arabic pay more per sentence and fit less conversation history in the same context window than a user writing English? Name the exact step where that cost is created.
