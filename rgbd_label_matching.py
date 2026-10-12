"""Small, inspectable known-label matcher using normalized Levenshtein distance."""

from dataclasses import dataclass
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def similarity(left, right):
    row = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1]+1, row[j]+1, row[j-1]+(a != b)))
        row = next_row
    return 1-row[-1]/max(len(left), len(right), 1)


@dataclass(frozen=True)
class LabelMatch:
    label: str
    score: float
    margin: float
    accepted: bool


class LabelMatcher:
    def __init__(self, labels, min_score=0.75, min_margin=0.15):
        if not labels or not 0 <= min_score <= 1 or not 0 <= min_margin <= 1:
            raise ValueError("Supply labels and score/margin thresholds in [0, 1]")
        self.labels = {name: [normalize(t) for t in [name, *aliases]] for name, aliases in labels.items()}
        self.min_score, self.min_margin = min_score, min_margin

    def match(self, text):
        word = normalize(text)
        # Short fragments such as single letters provide too little evidence.
        if len(word) < 4:
            return None
        scores = sorted(((max(similarity(word, alias) for alias in aliases), name)
                         for name, aliases in self.labels.items()), reverse=True)
        score, name = scores[0]
        margin = score-(scores[1][0] if len(scores) > 1 else 0)
        return LabelMatch(name, score, margin,
                          score >= self.min_score and margin >= self.min_margin)
