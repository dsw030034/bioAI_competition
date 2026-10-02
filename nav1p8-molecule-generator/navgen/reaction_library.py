"""Seed-linked amide recombination baseline; template routes are hypotheses."""
import argparse
import csv
import json
from pathlib import Path
import random
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from .generation import load_seed_smiles, candidate_id
from .config import load_config

FORWARD = "[C:1](=[O:2])[O;H1].[N;H1,H2;!$(N-C=O):3]>>[C:1](=[O:2])[N:3]"
RETRO = "[C:1](=[O:2])[N;H1,H2:3]>>[C:1](=[O:2])O.[N:3]"


def enumerate_library(seeds, blocks_path, seed=20260930):
    with open(blocks_path, encoding="utf-8-sig", newline="") as handle:
        blocks = list(csv.DictReader(handle))
    queries = {"acid": Chem.MolFromSmarts("C(=O)[O;H1]"), "amine": Chem.MolFromSmarts("[N;H1,H2;!$(N-C=O)]")}
    for block in blocks:
        mol = Chem.MolFromSmiles(block["smiles"])
        if block["role"] not in queries or mol is None or not mol.HasSubstructMatch(queries[block["role"]]):
            raise ValueError(f"Invalid block: {block}")
        block["mol"] = mol
    random.Random(seed).shuffle(blocks)
    retro = rdChemReactions.ReactionFromSmarts(RETRO)
    forward = rdChemReactions.ReactionFromSmarts(FORWARD)
    for parent in seeds:
        for acid, amine in retro.RunReactants((Chem.MolFromSmiles(parent),), maxProducts=100):
            Chem.SanitizeMol(acid)
            Chem.SanitizeMol(amine)
            for block in blocks:
                reactants = (block["mol"], amine) if block["role"] == "acid" else (acid, block["mol"])
                for products in forward.RunReactants(reactants, maxProducts=100):
                    try:
                        Chem.SanitizeMol(products[0])
                    except (ValueError, RuntimeError):
                        continue
                    mol = products[0]
                    for atom in mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                    smi = Chem.MolToSmiles(mol)
                    yield {"candidate_id": candidate_id(smi), "smiles": smi, "parent": parent,
                           "block_id": block["id"], "reaction_smarts": FORWARD,
                           "reactants": json.dumps([Chem.MolToSmiles(m) for m in reactants]),
                           "route_status": "template_only; availability/unvalidated"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--blocks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=300)
    a = p.parse_args()
    if a.budget < 1:
        p.error("budget must be positive")
    import itertools
    cfg = load_config(a.config)
    rows = list(itertools.islice(enumerate_library(load_seed_smiles(cfg["seed_file"]), a.blocks), a.budget))
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["candidate_id", "smiles", "parent", "block_id", "reaction_smarts", "reactants", "route_status"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"attempts": len(rows), "budget": a.budget, "space_exhausted": len(rows) < a.budget}))


if __name__ == "__main__":
    main()
