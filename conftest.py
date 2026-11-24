import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def trained():
    needed = [ROOT / "data/processed/panel.parquet", ROOT / "artifacts/forecaster.pt", ROOT / "artifacts/anomaly.pkl"]
    if not all(p.exists() for p in needed):
        pytest.skip("run `make data train` first")
    return True
