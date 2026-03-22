import os
import unittest
from pathlib import Path

class TestTestingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from src import testing as testing_module
        except Exception as exc:
            raise unittest.SkipTest(f"Cannot import src.testing: {exc}") from exc
        cls.testing_module = testing_module

    def _dataset_path(self) -> str | None:
        env_path = os.environ.get("ZEN_DATA_PATH")
        if env_path:
            return env_path
        cfg_path = self.testing_module.test_config.get("datapath")
        return str(cfg_path) if cfg_path else None

    def _all_model_names(self) -> tuple[str, ...]:
        models = ["stylecf", "transformer", "idm"]
        if "lstm_agent" in self.testing_module.test_config:
            models.append("lstm")
        return tuple(models)

    def test_run_testing_all_models_with_real_data(self):
        dataset_path = self._dataset_path()
        if not dataset_path or not os.path.exists(dataset_path):
            self.skipTest(f"Missing dataset: {dataset_path}")

        head = int(os.environ.get("TESTING_HEAD", "2000"))
        model_names = self._all_model_names()

        options = self.testing_module.TestingOptions(
            head=head,
            style_window=tuple(
                self.testing_module.test_config.get(
                    "style_window",
                    self.testing_module.DEFAULT_STYLE_WINDOW,
                )
            ),
            test_window=tuple(
                self.testing_module.test_config.get(
                    "test_window",
                    self.testing_module.DEFAULT_TEST_WINDOW,
                )
            ),
            start_time=int(self.testing_module.test_config.get("start_time", 60)),
            style_token_seconds=float(self.testing_module.test_config.get("style_token_seconds", 6.0)),
            style_token_mode=str(self.testing_module.test_config.get("style_token_mode", "per_sample")),
            output_dir=Path("models/test_results_unittest"),
            enabled_models=model_names,
            save_results=False,
            plot_results=False,
        )

        results = self.testing_module.run_testing(options)

        self.assertSetEqual(set(results.keys()), set(model_names))
        for model_name in model_names:
            result = results[model_name]
            self.assertGreater(
                len(result.metrics["MSE"]),
                0,
                f"{model_name} produced empty metric list",
            )
            self.assertGreater(
                len(result.errors),
                0,
                f"{model_name} produced no error sequences",
            )


if __name__ == "__main__":
    unittest.main()
