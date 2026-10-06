# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import string
from collections import Counter

ARTICLES = {"a", "an", "the"}
_STRIP_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    without_punctuation = s.lower().translate(_STRIP_PUNCTUATION)
    return " ".join(tok for tok in without_punctuation.split() if tok not in ARTICLES)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
