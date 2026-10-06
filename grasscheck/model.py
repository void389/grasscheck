from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z][a-z'-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize consistently without sending text off-device."""
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float


class OpenNaiveBayes:
    """Small, inspectable multinomial Naive Bayes text classifier."""

    def __init__(self, labels: list[str], priors: dict[str, float], counts: dict[str, dict[str, int]], totals: dict[str, int], vocabulary: list[str]):
        self.labels = labels
        self.priors = priors
        self.counts = counts
        self.totals = totals
        self.vocabulary = vocabulary

    @classmethod
    def train(cls, rows: list[tuple[str, str]]) -> "OpenNaiveBayes":
        label_docs: Counter[str] = Counter()
        word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()
        for text, label in rows:
            tokens = tokenize(text)
            if not tokens:
                continue
            label_docs[label] += 1
            word_counts[label].update(tokens)
            vocabulary.update(tokens)
        if len(label_docs) < 2:
            raise ValueError("training data must contain at least two labels")
        total_docs = sum(label_docs.values())
        labels = sorted(label_docs)
        return cls(
            labels=labels,
            priors={label: label_docs[label] / total_docs for label in labels},
            counts={label: dict(word_counts[label]) for label in labels},
            totals={label: sum(word_counts[label].values()) for label in labels},
            vocabulary=sorted(vocabulary),
        )

    def predict(self, text: str) -> list[Prediction]:
        tokens = tokenize(text)
        vocab_size = max(1, len(self.vocabulary))
        scores: dict[str, float] = {}
        for label in self.labels:
            score = math.log(self.priors[label])
            denominator = self.totals[label] + vocab_size
            label_counts = self.counts[label]
            for token in tokens:
                score += math.log((label_counts.get(token, 0) + 1) / denominator)
            scores[label] = score
        peak = max(scores.values())
        exponentials = {label: math.exp(score - peak) for label, score in scores.items()}
        denominator = sum(exponentials.values())
        return sorted(
            (Prediction(label, value / denominator) for label, value in exponentials.items()),
            key=lambda item: (-item.probability, item.label),
        )

    def to_dict(self) -> dict:
        return {
            "format": "grasscheck-naive-bayes-v1",
            "labels": self.labels,
            "priors": self.priors,
            "counts": self.counts,
            "totals": self.totals,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OpenNaiveBayes":
        if payload.get("format") != "grasscheck-naive-bayes-v1":
            raise ValueError("unsupported model format")
        return cls(
            labels=list(payload["labels"]),
            priors={k: float(v) for k, v in payload["priors"].items()},
            counts={label: {k: int(v) for k, v in words.items()} for label, words in payload["counts"].items()},
            totals={k: int(v) for k, v in payload["totals"].items()},
            vocabulary=list(payload["vocabulary"]),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OpenNaiveBayes":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_training_csv(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["text"], row["label"]) for row in csv.DictReader(handle)]
