try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path

from navgen.reinvent import prepare_reinvent_config
from navgen.config import FilterConfig
from navgen.scoring import reinvent_native_metrics


def test_reinvent_configs_use_current_schema(tmp_path):
    result = prepare_reinvent_config(
        tmp_path / "reinvent",
        "models/mol2mol_medium_similarity.prior",
        project_config=Path("configs/site_c_reference.json"),
    )
    sampling = tomllib.loads(Path(result["sampling"]).read_text(encoding="utf-8"))
    learning = tomllib.loads(Path(result["staged_learning"]).read_text(encoding="utf-8"))
    assert sampling["run_type"] == "sampling"
    assert sampling["parameters"]["model_file"].endswith("mol2mol_medium_similarity.prior")
    assert learning["run_type"] == "staged_learning"
    assert learning["parameters"]["agent_file"].endswith("mol2mol_medium_similarity.prior")
    assert learning["stage"][0]["scoring"]["type"] == "geometric_mean"


def test_local_reinvent_metrics_are_bounded():
    metrics = reinvent_native_metrics({"mw": 350.0, "logp": 2.0, "qed": 0.8}, FilterConfig())
    assert 0.0 <= metrics["reinvent_geometric_mean"] <= 1.0
    assert metrics["reinvent_geometric_mean"] > 0.0
