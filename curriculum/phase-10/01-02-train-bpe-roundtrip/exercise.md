# Train BPE and round-trip

Topic: 1. Tokenization and BPE
Difficulty: 2 of 3

## Problem

Train a byte-level BPE tokenizer and use it to encode and decode text. Pure Python only.

- `train_bpe(text: str, vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]`. Start from the UTF-8 bytes of `text` (IDs 0–255). Repeat until the vocab has `vocab_size` entries: count adjacent pairs, take the most frequent (ties: the pair first seen scanning left to right), give it the next free ID (256, 257, …) and merge it everywhere. Return `merges` (pair → new ID, in the order they were learned) and `vocab` (ID → bytes, including all 256 single bytes). Stop early if no pair occurs at least twice.
- `encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]` turns text into bytes, then applies every learned merge in the order it was learned.
- `decode(ids: list[int], vocab: dict[int, bytes]) -> str` concatenates the bytes and decodes as UTF-8, replacing invalid sequences (`errors="replace"`).

`vocab_size` is at least 256.

## Examples

```
merges, vocab = train_bpe("aaabdaaabac", 259)
merges          → {(97, 97): 256, (256, 97): 257, (257, 98): 258}
vocab[256]      → b"aa"
vocab[258]      → b"aaab"
encode("aaabdaaabac", merges)          → [258, 100, 258, 97, 99]
decode([258, 100, 258, 97, 99], vocab) → "aaabdaaabac"
```

## Constraints

- `text` is up to 5 000 characters; `vocab_size` is 256 to 512.
- Any Unicode text must round-trip: `decode(encode(t))` is `t` for accents, Arabic script and emoji.
- Pure Python only.

## Hints

1. Which function from the previous exercise gives you the most frequent pair, and which one applies it?
2. If a pair was learned as the 3rd merge, can its two halves exist in the sequence before merges 1 and 2 ran? What does that say about the order `encode` must apply merges in?
3. How many bytes does "🌍" occupy in UTF-8? What does `vocab[256..]` need to store so `decode` can rebuild them?
4. What happens when `vocab_size` asks for more merges than the text can supply?

## Explain-back

- Does the model ever see letters? What does it see for "strawberry", and why can it miscount the r's?
- Why does a bigger vocab make sequences shorter but the embedding table larger? Where is the sweet spot for a 100k-vocab model?
- Can every list of token IDs be decoded to valid text? Show an ID list that cannot, and explain what `errors="replace"` does with it.
- Why would a real tokenizer pre-split on a regex (words, numbers, punctuation) before counting pairs?
