"""Run the full evaluation suite -> artifacts/eval_report.{md,json}."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.ERROR)

from surgeshield.config import path  # noqa: E402
from surgeshield.eval.harness import run_all  # noqa: E402

if __name__ == "__main__":
    r = run_all()
    print(path("artifacts/eval_report.md").read_text())
    sys.exit(0 if r["scenario_pass_rate"] == 1.0 else 1)
