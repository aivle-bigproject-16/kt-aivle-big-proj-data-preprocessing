import unittest

from battery_v4_1 import __version__


class VersionTests(unittest.TestCase):
    def test_runtime_version_matches_project_version(self):
        self.assertEqual(__version__, "4.1.0")


if __name__ == "__main__":
    unittest.main()
