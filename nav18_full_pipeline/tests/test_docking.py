import json
from pathlib import Path

import pytest

from navgen.docking import DockingConfig


def test_docking_config_requires_real_inputs(tmp_path):
    config = tmp_path / "docking.json"
    config.write_text(json.dumps({
        "vina_executable": "vina.exe",
        "receptor_pdbqt": "receptor.pdbqt",
        "ligand_dir": "ligands",
        "output_dir": "out",
        "center": [1, 2, 3],
        "size": [20, 20, 20]
    }), encoding="utf-8")
    parsed = DockingConfig.from_json(config)
    with pytest.raises(FileNotFoundError):
        from navgen.docking import run_vina_docking
        run_vina_docking(parsed)
