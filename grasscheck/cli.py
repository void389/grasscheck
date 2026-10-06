from __future__ import annotations

import argparse
from importlib.resources import files

from .model import OpenNaiveBayes
from .planner import load_activities, recommend


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Get a tiny offline AI nudge to leave the screen.")
    result.add_argument("feeling", help="How you feel or what you want from going outside")
    result.add_argument("--minutes", type=int, default=30)
    result.add_argument("--energy", choices=["low", "medium", "high"], default="medium")
    result.add_argument("--weather", choices=["clear", "cloudy", "drizzle", "snow", "hot"], default="clear")
    result.add_argument("--setting", choices=["urban", "suburban", "rural", "park"], default="urban")
    result.add_argument("--accessible", action="store_true", help="Only suggest step-free activities")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    package = files("grasscheck")
    model = OpenNaiveBayes.load(package / "data" / "model.json")
    activities = load_activities(package / "data" / "activities.json")
    choices = recommend(model, activities, args.feeling, minutes=args.minutes, energy=args.energy, weather=args.weather, setting=args.setting, accessible=args.accessible)
    if not choices:
        print("No exact match. Try allowing more time, energy, or another nearby setting.")
        return 2
    print("\nGRASSCHECK - pick one, then close the screen\n")
    for index, choice in enumerate(choices, 1):
        print(f"{index}. {choice.activity.name} - {choice.activity.minutes} min")
        print(f"   {choice.activity.prompt}")
        print(f"   Why: {choice.reason}\n")
    print("Your move: choose 1-3, take what you need, and go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
