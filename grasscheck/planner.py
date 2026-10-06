from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import OpenNaiveBayes


@dataclass(frozen=True)
class Activity:
    name: str
    category: str
    minutes: int
    energy: int
    weather: tuple[str, ...]
    settings: tuple[str, ...]
    accessible: bool
    prompt: str


@dataclass(frozen=True)
class Recommendation:
    activity: Activity
    score: float
    reason: str


ENERGY = {"low": 1, "medium": 2, "high": 3}


def load_activities(path: Path) -> list[Activity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Activity(
            name=item["name"],
            category=item["category"],
            minutes=int(item["minutes"]),
            energy=ENERGY[item["energy"]],
            weather=tuple(item["weather"]),
            settings=tuple(item["settings"]),
            accessible=bool(item["accessible"]),
            prompt=item["prompt"],
        )
        for item in payload
    ]


def recommend(model: OpenNaiveBayes, activities: list[Activity], text: str, *, minutes: int, energy: str, weather: str, setting: str, accessible: bool = False, limit: int = 3) -> list[Recommendation]:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if energy not in ENERGY:
        raise ValueError(f"energy must be one of: {', '.join(ENERGY)}")
    probabilities = {item.label: item.probability for item in model.predict(text)}
    matches: list[Recommendation] = []
    for activity in activities:
        if activity.minutes > minutes or activity.energy > ENERGY[energy]:
            continue
        if weather not in activity.weather or setting not in activity.settings:
            continue
        if accessible and not activity.accessible:
            continue
        time_fit = 1 - abs(minutes - activity.minutes) / max(minutes, activity.minutes)
        energy_fit = 1 - abs(ENERGY[energy] - activity.energy) / 3
        score = 0.68 * probabilities.get(activity.category, 0) + 0.18 * time_fit + 0.14 * energy_fit
        reason = f"{round(probabilities.get(activity.category, 0) * 100)}% model match; fits {activity.minutes} min and {energy} energy"
        matches.append(Recommendation(activity, score, reason))
    return sorted(matches, key=lambda item: (-item.score, item.activity.minutes, item.activity.name))[:limit]
