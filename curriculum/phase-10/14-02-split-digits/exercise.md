# Split the digits before you merge

Topic: 14. Tokenization revisited and LLM quirks
Difficulty: 2 of 3

## Problem

Left alone, BPE learns whatever digit blobs the training text happened to contain — `"1234"` as one token, `"567"` as another — so the same digit sits in a different token depending on its neighbours, and column-wise arithmetic has nothing stable to line up. Real tokenizers therefore **pre-split** numbers before any merge runs. Build that rule and measure what it costs. Pure Python only.

Write two functions.

`split_digits(text: str) -> list[str]` cuts `text` into chunks that a merge is never allowed to cross:

- A **digit** is one of `0123456789` only — not `"²"`, not Eastern-Arabic numerals, so do not use `str.isdigit()`.
- Scan left to right. A maximal run of digits becomes groups of **at most 3, counted from the right end of the run**, so only the *first* group of a run may be shorter than 3. `"1234567"` → `"1"`, `"234"`, `"567"`.
- A maximal run of non-digit characters becomes one chunk, unchanged.
- Chunks come out in order and `"".join(split_digits(text)) == text` always holds. `split_digits("")` is `[]`.

`token_count_delta(text: str, merges: dict[tuple[int, int], int]) -> int` returns **how many extra tokens the rule costs**:

```
token_count_delta = tokens_with_the_split - tokens_without_the_split
```

Both counts use the same byte-level scheme: take the UTF-8 bytes of a string (each byte an ID in 0–255), then walk `merges` in the dict's own order (its insertion order, which is the order the merges were learned) and for each `(a, b) -> new_id` make one left-to-right pass replacing every non-overlapping adjacent `a, b` with `new_id`.

- *Without* the split: encode `text` as one string and count the IDs.
- *With* the split: encode **each chunk of `split_digits(text)` on its own** and add the counts up, so no merge can join IDs from two different chunks.

Never modify `merges`. The result is an `int`; it is 0 when no merge was spanning a chunk boundary anyway, and positive when the split broke one up.

## Examples

```
split_digits("1234567")        → ["1", "234", "567"]        groups from the RIGHT, not ["123", "456", "7"]
split_digits("abc123")         → ["abc", "123"]
split_digits("a1b22c333d4444") → ["a", "1", "b", "22", "c", "333", "d", "4", "444"]
split_digits("2024-01-02")     → ["2", "024", "-", "01", "-", "02"]
split_digits("1000000")        → ["1", "000", "000"]
split_digits("007")            → ["007"]
split_digits("no digits here") → ["no digits here"]
split_digits("")               → []

MERGES = {(116, 49): 256, (49, 50): 257, (51, 52): 258, (257, 258): 259, (53, 54): 260}
#              "t1"           "12"           "34"          "1234"            "56"

token_count_delta("1234567", MERGES)        → 2    plain [259, 260, 55] = 3; split 1 + 2 + 2 = 5
token_count_delta("t1234", MERGES)          → 1    plain [256, 50, 258] = 3; split 1 + 1 + 2 = 4
token_count_delta("abc123", MERGES)         → 0    no merge spanned a boundary
token_count_delta("no digits here", MERGES) → 0
token_count_delta("", MERGES)               → 0
```

## Constraints

- `text` is at most 5 000 characters and may contain any Unicode; only ASCII `0`–`9` count as digits.
- `merges` has at most 512 entries, in learned order, and is never modified.
- `split_digits` returns a `list[str]` whose concatenation is exactly `text`; `token_count_delta` returns an `int`.
- Pure Python only — no regex library is required and nothing is downloaded.

## Hints

1. `"1234"` is four digits and `"12345"` is five. If every group but one must hold exactly 3, which end of the run do you have to start counting from, and which group is the odd-sized one?
2. Write down the digits of 1234567 under the digits of 7654321, as you would to add them by hand. Which grouping — from the left or from the right — keeps the ones column under the ones column?
3. Your two counts must come from the *same* encoder. What is the only difference between "encode the whole string" and "encode each chunk"? Where does a merge that used to span a boundary now have nothing to grab?
4. Why would the delta ever be exactly 0? What has to be true of `merges` and of this particular text for the split to change nothing?

## Explain-back

- Trace why an LLM can multiply small numbers but fumbles long ones. What does the model actually receive for `"1234567"` under a tokenizer with no digit rule, and what does it have to learn that a human doing long multiplication never has to?
- The digit pre-split makes prompts containing numbers cost *more* tokens. Why do tokenizer designers accept that bill?
- Why does grouping from the right rather than the left matter for arithmetic specifically? What property of place value is being preserved?
- The same model is asked to reverse `"strawberry"` and to spell it out letter by letter. Both are easy for you and awkward for it. What does its input look like for that word, and what would have to change in the tokenizer to make the task easy?
