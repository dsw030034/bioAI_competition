"""Traceable virtual building blocks and acid x amine enumeration, without route claims."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions


RETRO = "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]"
FORWARD = "[C:1](=[O:2])[O;H1;+0].[N;H1,H2;+0;!$(N-C=O);!$(N-S(=O)=O);!$(N-P(=O)):3]>>[C:1](=[O:2])[N:3]"
ROLE_QUERIES = {"acid": "[CX3](=O)[OX2H1;+0]", "amine": "[N;H1,H2;+0;!$(N-C=O);!$(N-S(=O)=O);!$(N-P(=O))]"}


def clean_mol(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    return mol


def build_blocks(original_csv, candidates, allowed_scopes):
    pool, diagnostics = {}, []
    def add(role, mol, source):
        if mol is None:
            raise ValueError("Invalid block molecule")
        mol = clean_mol(mol)
        if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            raise ValueError("Block must be a single component without dummy atoms")
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ROLE_QUERIES[role]))
        if not matches:
            raise ValueError(f"No compatible {role} handle")
        smiles = Chem.MolToSmiles(mol)
        key = (role, smiles)
        row = pool.setdefault(key, {"block_id": role + "_" + hashlib.sha256(smiles.encode()).hexdigest()[:12],
                                   "role": role, "smiles": smiles, "sources": [], "reactive_handle_count": len(matches),
                                   "availability": "not_verified", "route_status": "virtual_block; synthesis_unvalidated"})
        if source not in row["sources"]:
            row["sources"].append(source)
    with Path(original_csv).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            add(row["role"], Chem.MolFromSmiles(row["smiles"]), {"type": "existing_block", "block_id": row["id"]})
    retro = rdChemReactions.ReactionFromSmarts(RETRO)
    for row in candidates:
        if row["scope"] not in allowed_scopes:
            continue
        products = retro.RunReactants((Chem.MolFromSmiles(row["smiles"]),), maxProducts=100)
        if len(products) >= 100:
            raise ValueError("Retrosplit cap reached; review source before truncating")
        for split_index, pair in enumerate(products):
            for role, mol in zip(("acid", "amine"), pair):
                try:
                    add(role, mol, {"type": "retrosplit_reviewed_candidate", "candidate_id": row["candidate_id"],
                                   "scope": row["scope"], "split_index": split_index, "reaction_smarts": RETRO})
                except (ValueError, RuntimeError) as exc:
                    diagnostics.append({"candidate_id": row["candidate_id"], "split_index": split_index,
                                        "role": role, "status": "block_rejected", "error": str(exc)})
    return sorted(pool.values(), key=lambda row: row["block_id"]), diagnostics


def cross_proposals(blocks, seed_smiles):
    acid = [r for r in blocks if r["role"] == "acid"]
    amine = [r for r in blocks if r["role"] == "amine"]
    reaction = rdChemReactions.ReactionFromSmarts(FORWARD)
    for a, b in itertools.product(acid, amine):
        base = {"parent_smiles": seed_smiles, "acid_block_id": a["block_id"], "amine_block_id": b["block_id"],
                "reactants": [a["smiles"], b["smiles"]], "reaction_smarts": FORWARD,
                "pair_id": a["block_id"] + ":" + b["block_id"],
                "reactive_handle_counts": [a["reactive_handle_count"], b["reactive_handle_count"]],
                "route_status": "template_only; availability_selectivity_conditions_unvalidated",
                "parent_relation": "root_series_reference; both reactants may be candidate-derived"}
        try:
            products = reaction.RunReactants((Chem.MolFromSmiles(a["smiles"]), Chem.MolFromSmiles(b["smiles"])), maxProducts=100)
            if len(products) >= 100:
                raise ValueError("Product cap reached; refusing silent truncation")
            if not products:
                yield {**base, "smiles": "", "generation_error": "no_reaction_product"}
            for index, product in enumerate(products):
                try:
                    mol = clean_mol(product[0])
                    yield {**base, "product_index": index, "smiles": Chem.MolToSmiles(mol)}
                except (ValueError, RuntimeError) as exc:
                    yield {**base, "product_index": index, "smiles": "", "generation_error": str(exc)}
        except (ValueError, RuntimeError) as exc:
            yield {**base, "smiles": "", "generation_error": str(exc)}
