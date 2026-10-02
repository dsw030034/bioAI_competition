from copy import deepcopy
from pathlib import Path

import pytest

from navgen.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    c = deepcopy(load_config(ROOT / "configs/demo.json"))
    c["output_dir"] = str(tmp_path / "batch")
    c["conformers"]["enabled"] = False
    return c

