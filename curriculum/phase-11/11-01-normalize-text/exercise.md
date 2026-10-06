# Text normalization for TTS

Topic: 11. Text-to-speech
Difficulty: 1 of 3

## Problem

Before a TTS model sees anything, the text front end rewrites it into words a speaker would actually say: "Dr." becomes "Doctor", "3" becomes "three". Everything downstream — the phonemizer, the duration predictor, the acoustic model — reads that rewritten string, so a mistake here is a mistake you hear.

Pure Python, no regex library needed (`re` is allowed but not required), no numpy.

- `number_to_words(n: int) -> str` for `0 <= n <= 999`, raising `ValueError` outside that range:
  - 0–19: `zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen`.
  - 20–99: `twenty thirty forty fifty sixty seventy eighty ninety`, joined to a non-zero remainder with a hyphen: `21` → `twenty-one`, `90` → `ninety`.
  - 100–999: the hundreds digit, then `hundred`, then a space and the remainder if it is non-zero, with **no** "and": `100` → `one hundred`, `115` → `one hundred fifteen`, `342` → `three hundred forty-two`.
- `normalize_text(s: str) -> str`:
  1. Split `s` on whitespace (so any run of spaces, tabs or newlines becomes one space and leading/trailing space disappears).
  2. Rewrite each token independently, in order:
     - If the **whole token** is a key of the abbreviation table below, replace it with its expansion. The match is exact and case-sensitive: `Dr.` matches, `dr.` and `Dr.J` do not.
     - Otherwise strip the longest suffix of `.,!?;:` characters off the token, call the rest `core` and the suffix `tail`. If `core` is non-empty and every character of it is a digit, replace the token with `number_to_words(int(core)) + tail`.
     - Otherwise leave the token exactly as it is.
  3. Join the rewritten tokens with single spaces.

  Abbreviation table: `Dr.`→`Doctor`, `Mr.`→`Mister`, `Mrs.`→`Missus`, `Ms.`→`Miss`, `Prof.`→`Professor`, `St.`→`Saint`, `Jr.`→`Junior`, `No.`→`Number`, `vs.`→`versus`, `etc.`→`et cetera`, `approx.`→`approximately`.

  A token that is a number outside 0–999 propagates the `ValueError` from `number_to_words`.

The rewrite is per token, never a substring replacement: `13` must not become `one three`, and `3rd` must come out unchanged.

## Examples

```
normalize_text("Dr. 3")                        → "Doctor three"
normalize_text("Mr. Smith has 21 cats.")       → "Mister Smith has twenty-one cats."
normalize_text("No. 115, approx. 7 km")        → "Number one hundred fifteen, approximately seven km"
normalize_text("  wide   spacing\n here ")     → "wide spacing here"
normalize_text("13 vs. 3rd")                   → "thirteen versus 3rd"
normalize_text("St. Ives etc.")                → "Saint Ives et cetera"
normalize_text("1000 cats")                    → ValueError

number_to_words(0)    → "zero"
number_to_words(20)   → "twenty"
number_to_words(42)   → "forty-two"
number_to_words(100)  → "one hundred"
number_to_words(342)  → "three hundred forty-two"
number_to_words(-1)   → ValueError
```

## Constraints

- Input strings up to 10 000 characters. Output compared exactly, character for character.
- Pure Python. No downloads, no language model, no number-spelling library (`inflect`, `num2words` and friends are banned — write the tables).

## Hints

1. Which three ranges does `number_to_words` really have to handle separately, and which one of them can call the function again on a smaller number?
2. `"cats."` and `"21."` both end in a period. What is the one rule that strips the period from the number but leaves `"cats."` alone?
3. Why check the abbreviation table before stripping punctuation off the token, rather than after?
4. `"13"` run through a naive `replace("3", "three")` gives `"1three"`. What does working token by token, on the whole token, buy you — and which of the examples above would a substring replace get wrong?

## Explain-back

- "TTS is just reading phonemes aloud." Name two things this function decided that no phoneme table could decide, and one it got wrong because it cannot see the sentence.
- `St.` is `Saint` in `St. Ives` and `Street` in `Baker St.`. What information settles it, and where in a modern TTS stack would that decision live?
- `1996` could be "one thousand nine hundred ninety-six", "nineteen ninety-six" or "one nine nine six". What would you need to know to choose, and what does getting it wrong do to the duration predictor downstream?
- Some end-to-end systems feed raw characters straight into the model and skip this step. What do they gain, and which failure mode do they take on?
