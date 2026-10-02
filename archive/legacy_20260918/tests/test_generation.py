from rdkit import Chem
from rdkit.Chem import rdChemReactions

from navgen.chemistry import canonical, molecule
from navgen.generation import AMIDE_FORWARD, proposals, read_seeds
from navgen.scoring import Scorer, interval_score


def test_seed_identity_and_stereochemistry(config):
    seeds = read_seeds(config["seeds_file"])
    scored = Scorer(config, seeds).score(seeds[0])
    assert scored["similarity"] == 1
    assert scored["reasons"] == ["seed_identity"]
    assert canonical("C[C@H](O)F") != canonical("C[C@@H](O)F")
    assert canonical("[CH3:42]O") == "CO"


def test_template_routes_replay_and_preserve_one_seed_fragment(config):
    seeds = read_seeds(config["seeds_file"])
    generated = list(proposals(config, seeds))
    assert len(generated) >= 30
    forward = rdChemReactions.ReactionFromSmarts(AMIDE_FORWARD)
    for proposal in generated:
        assert molecule(proposal.smiles) is not None
        replayed = forward.RunReactants(tuple(molecule(s) for s in proposal.route["reactants"]))
        assert proposal.smiles in {canonical(Chem.MolToSmiles(p[0])) for p in replayed}
        assert proposal.parent_smiles == seeds[0]
        assert proposal.route["validation"] == "template_only"


def test_two_sided_membrane_constraint():
    assert interval_score(2.5, 1, 4) == 1
    assert interval_score(-2, 1, 4) == interval_score(7, 1, 4) < 1
    assert interval_score(10, 1, 4) < interval_score(7, 1, 4)


def test_invalid_salts_wildcards_and_missing_scores(config):
    scorer = Scorer(config, read_seeds(config["seeds_file"]))
    for raw in ("", "not-a-smiles", "C1CC", "[CH5]"):
        assert scorer.score(raw)["valid"] is False
    assert "disconnected_structure" in scorer.score("CC.[Na+]")["reasons"]
    assert "unsupported_element" in scorer.score("C*")["reasons"]
    good = next(p for p in proposals(config, scorer.seeds) if scorer.score(p.smiles)["passed"])
    row = scorer.score(good.smiles)
    assert row["ddg_pred"] is None
    assert row["selectivity_status"] == "not_available"
    assert 0 < row["reward"] <= 1


def test_required_substructure(config):
    config["filters"]["required_smarts"] = "[Si]"
    scorer = Scorer(config, read_seeds(config["seeds_file"]))
    assert "required_substructure" in scorer.score(scorer.seeds[0])["reasons"]

