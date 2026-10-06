# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
ABBREVIATIONS = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Prof.": "Professor",
    "St.": "Saint",
    "Jr.": "Junior",
    "No.": "Number",
    "vs.": "versus",
    "etc.": "et cetera",
    "approx.": "approximately",
}
PUNCTUATION = ".,!?;:"


def number_to_words(n: int) -> str:
    if not 0 <= n <= 999:
        raise ValueError("number_to_words handles integers 0..999")
    if n < 20:
        return ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return TENS[tens - 2] + (f"-{ONES[rest]}" if rest else "")
    hundreds, rest = divmod(n, 100)
    return f"{ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")


def normalize_text(s: str) -> str:
    out = []
    for token in s.split():
        if token in ABBREVIATIONS:
            out.append(ABBREVIATIONS[token])
            continue
        core = token.rstrip(PUNCTUATION)
        tail = token[len(core):]
        if core and all(ch.isdigit() for ch in core):
            out.append(number_to_words(int(core)) + tail)
        else:
            out.append(token)
    return " ".join(out)
