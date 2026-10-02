from navgen.evaluation import evaluate_library


def test_evaluate_library_basic_metrics():
    smiles = [
        "O=C(Nc1ccccc1)c1ccccc1",
        "O=C(Nc1ccc(Cl)cc1)c1ccccc1",
        "O=C(Nc1ccccc1)c1ccco1",
        "not-a-smiles",
        "O=C(Nc1ccccc1)c1ccccc1",  # duplicate
    ]
    metrics = evaluate_library(smiles)
    assert metrics["n_raw"] == 5
    assert metrics["n_valid"] == 4
    assert metrics["n_unique"] == 3
    assert metrics["validity"] == 0.8
    assert metrics["uniqueness"] == 0.75
    assert metrics["n_murcko_scaffolds"] >= 1
    assert metrics["distribution"] is not None
    assert metrics["reference_novelty"] is None


def test_evaluate_library_reference_novelty():
    smiles = [
        "O=C(Nc1ccccc1)c1ccccc1",
        "O=C(Nc1ccc(Cl)cc1)c1ccccc1",
    ]
    reference = ["O=C(Nc1ccccc1)c1ccccc1"]
    metrics = evaluate_library(smiles, reference_smiles=reference)
    assert metrics["reference_novelty"] is not None
    assert "fraction_novel_lt_0_4" in metrics["reference_novelty"]


def test_evaluate_library_intra_similarity():
    smiles = ["O=C(Nc1ccccc1)c1ccccc1", "O=C(Nc1ccccc1)c1ccco1"]
    metrics = evaluate_library(smiles)
    assert metrics["intra_batch_tanimoto_mean"] is not None
