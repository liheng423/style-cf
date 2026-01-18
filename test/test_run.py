import unittest


if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromName("idm.TestIDMAgent"))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName("stylecf.TestStyleAgent"))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
