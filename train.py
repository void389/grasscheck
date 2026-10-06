from pathlib import Path

from grasscheck.model import OpenNaiveBayes, load_training_csv


root = Path(__file__).parent
model = OpenNaiveBayes.train(load_training_csv(root / "grasscheck" / "data" / "training.csv"))
model.save(root / "grasscheck" / "data" / "model.json")
print(f"trained {len(model.labels)} labels with {len(model.vocabulary)} visible token weights")
