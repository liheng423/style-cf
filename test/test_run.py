import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromName(
            "test.test_idm_calibrate.TestIdmCalibrate"
        )
    )
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromName("test.idm.TestIDMAgent"))
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromName("test.stylecf.TestStyleAgent"))
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromName(
            "test.test_testing_pipeline.TestTestingPipeline"
        )
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
