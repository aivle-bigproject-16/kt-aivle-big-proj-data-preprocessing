import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_cli_help_exposes_safety_workflow(self):
        result = subprocess.run(
            [sys.executable, "-m", "battery_v3_9.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertIn("approve-selection", result.stdout)
        self.assertIn("execute", result.stdout)


if __name__ == "__main__":
    unittest.main()

