# GrassCheck

GrassCheck is a tiny, transparent, offline AI that turns a sentence about how you feel into three realistic reasons to close the screen and go outside.

It was created during the **Hacktoberfest Open-Source AI Challenge: Week 1 — Touch Grass** (October 5–11, 2026). The project deliberately uses a small model and a short interaction: the screen should be the shortest part of an outdoor experience.

## Try it

Python 3.10+ is the only requirement.

```console
python -m grasscheck "I am restless and need to move" --minutes 25 --energy medium --weather cloudy --setting urban
```

Example:

```text
GRASSCHECK - pick one, then close the screen

1. Block interval walk - 20 min
   Alternate one block brisk and one block easy. Turn around after five cycles.
   Why: 63% model match; fits 20 min and medium energy

Your move: choose 1–3, take what you need, and go.
```

Use `--accessible` to keep only step-free activities. Weather choices are `clear`, `cloudy`, `drizzle`, `snow`, and `hot`; settings are `urban`, `suburban`, `rural`, and `park`.

## The open AI

GrassCheck uses a multinomial Naive Bayes text model trained on 48 plain-English examples across six intentions: notice, move, make, care, connect, and explore. The implementation uses only Python's standard library.

Everything is inspectable:

- `grasscheck/data/training.csv` contains every training example.
- `train.py` recreates the model.
- `grasscheck/data/model.json` contains every learned count and prior.
- `grasscheck/model.py` is the complete training and inference implementation.
- `grasscheck/data/activities.json` contains the constrained activity catalog.

No prompt, feeling, location, or preference leaves the device. There is no account, API key, network request, analytics event, or paid inference endpoint.

The model predicts the user's intent. A deterministic planner then rejects activities that do not fit the available time, energy, weather, setting, or accessibility need, and ranks the remaining choices using the model probability plus constraint fit. This split keeps the model central while making safety and feasibility rules easy to audit.

## Train and verify

```console
python train.py
python -m unittest discover -s tests -v
```

The model file is committed so a fresh checkout works immediately. Re-training is deterministic and produces the same JSON.

## Design principles

1. **One sentence in, three exits out.** No feed and no engagement loop.
2. **Open weights should be readable.** A curious user can inspect the learned counts with a text editor.
3. **Constraints beat wishful thinking.** A ten-minute drizzle plan should not recommend a forty-minute trail mission.
4. **Privacy is easier when inference is local.** Feelings and mobility preferences remain on the user's machine.
5. **The product succeeds when it is closed.** The final line asks the user to choose and leave.

## Safety

GrassCheck suggests low-stakes activities from a fixed catalog. Users should follow local rules, weather alerts, accessibility conditions, and their own judgment. It does not provide medical, route, or emergency advice.

## License

MIT
