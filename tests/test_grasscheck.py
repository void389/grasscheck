import tempfile
import unittest
from pathlib import Path

from grasscheck.model import OpenNaiveBayes, load_training_csv, tokenize
from grasscheck.planner import load_activities, recommend


ROOT = Path(__file__).parents[1]
DATA = ROOT / "grasscheck" / "data"


class GrassCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = OpenNaiveBayes.train(load_training_csv(DATA / "training.csv"))
        cls.activities = load_activities(DATA / "activities.json")

    def test_tokenizer_is_local_and_deterministic(self):
        self.assertEqual(tokenize("Calm, screen-free WALK!"), ["calm", "screen-free", "walk"])

    def test_classifier_finds_movement_intent(self):
        self.assertEqual(self.model.predict("I am restless and need exercise")[0].label, "move")

    def test_classifier_finds_care_intent(self):
        self.assertEqual(self.model.predict("I want to help clean my park")[0].label, "care")

    def test_model_round_trip_keeps_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            restored = OpenNaiveBayes.load(path)
            self.assertEqual(restored.predict("quiet mindful break"), self.model.predict("quiet mindful break"))

    def test_recommendations_obey_time_weather_and_setting(self):
        results = recommend(self.model, self.activities, "I want a peaceful pause", minutes=15, energy="low", weather="drizzle", setting="urban")
        self.assertTrue(results)
        self.assertTrue(all(item.activity.minutes <= 15 for item in results))
        self.assertTrue(all("drizzle" in item.activity.weather for item in results))
        self.assertTrue(all("urban" in item.activity.settings for item in results))

    def test_accessible_filter_is_strict(self):
        results = recommend(self.model, self.activities, "I want an adventure", minutes=60, energy="high", weather="clear", setting="park", accessible=True, limit=20)
        self.assertTrue(results)
        self.assertTrue(all(item.activity.accessible for item in results))

    def test_invalid_minutes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            recommend(self.model, self.activities, "anything", minutes=0, energy="low", weather="clear", setting="park")

    def test_same_input_has_same_rank_order(self):
        kwargs = dict(minutes=30, energy="medium", weather="clear", setting="park")
        first = recommend(self.model, self.activities, "creative nature project", **kwargs)
        second = recommend(self.model, self.activities, "creative nature project", **kwargs)
        self.assertEqual([item.activity.name for item in first], [item.activity.name for item in second])


if __name__ == "__main__":
    unittest.main()
