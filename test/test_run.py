import unittest


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromName("idm.TestIDMAgent")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
