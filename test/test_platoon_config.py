import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.platoon.config_loader import get_platoon_configs


class TestPlatoonConfig(unittest.TestCase):
    def test_load_platoon_configs(self):
        cfg = get_platoon_configs(force_reload=True)
        self.assertIn("simulation_config", cfg)
        self.assertIn("newell_config", cfg)
        self.assertIn("evaluation_config", cfg)
        self.assertIn("experiments", cfg)

    def test_append_and_summary(self):
        try:
            from src.platoon.runner import append_results, summarize_group_metrics
        except Exception as exc:
            self.skipTest(f"platoon runner dependencies unavailable: {exc}")

        results = {}
        append_results(results, {"delay": 10.0, "flow": [1.0, 2.0], "density": []})
        append_results(results, {"delay": 20.0, "flow": [3.0]})
        self.assertEqual(results["delay"], [10.0, 20.0])
        self.assertEqual(results["flow"], [1.0, 2.0, 3.0])

        summary = summarize_group_metrics({"plat1": results})
        self.assertEqual(summary.shape[0], 1)
        self.assertIn("delay", summary.columns)


if __name__ == "__main__":
    unittest.main()
