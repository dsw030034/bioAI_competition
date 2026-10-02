from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable, Iterable

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:  # pragma: no cover - fallback for minimal environments
    Chem = None
    AllChem = None
    DataStructs = None

DEFAULT_BLOCKS = [
    "CCN",
    "CCCN",
    "NCCO",
    "COC1=CC=CC=C1",
    "FC1=CC=CC=C1",
    "ClC1=CC=CC=C1",
    "CC(C)N",
    "C1=CC=CC=C1",
    "COCCN",
    "c1ccccc1O",
]


def canonical_smiles(smiles: str) -> str:
    if Chem is None:
        return smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def candidate_id(smiles: str) -> str:
    return hashlib.sha256(canonical_smiles(smiles).encode("utf-8")).hexdigest()[:16]


def load_seed_smiles(path: str | Path) -> list[str]:
    seed_file = Path(path)
    seeds: list[str] = []
    if not seed_file.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smiles = line.split()[0]
        if Chem is not None and Chem.MolFromSmiles(smiles) is None:
            continue
        seeds.append(canonical_smiles(smiles))
    if not seeds:
        raise ValueError(f"No valid seed molecules found in {seed_file}")
    return seeds


def has_amide_motif(mol: Chem.Mol | str) -> bool:
    if Chem is None:
        candidate = str(mol)
        return "C(=O)N" in candidate or "NC(=O)" in candidate
    pattern = Chem.MolFromSmarts("[#6](=[#8])-[#7]")
    return bool(mol.HasSubstructMatch(pattern))


def build_replacement_candidates(seed_smiles: str, blocks: Iterable[str] | None = None, max_candidates: int = 30) -> list[dict]:
    seed = canonical_smiles(seed_smiles)
    if Chem is not None:
        mol = Chem.MolFromSmiles(seed)
        if mol is None:
            raise ValueError(f"Invalid seed SMILES: {seed}")
    else:
        mol = seed

    if Chem is not None:
        analogs = _build_seed_analogs(seed, mol, max_candidates)
        if analogs:
            return analogs

    block_list = list(blocks or DEFAULT_BLOCKS)
    random.Random(0).shuffle(block_list)
    block_list = block_list[:max_candidates]

    records: list[dict] = []
    for idx, block in enumerate(block_list):
        if Chem is not None and Chem.MolFromSmiles(block) is None:
            continue

        candidate_smiles = f"{block}C(=O)Nc1ccccc1"
        candidate_smiles = canonical_smiles(candidate_smiles)
        if candidate_smiles == seed:
            continue
        records.append(
            {
                "candidate_id": candidate_id(candidate_smiles),
                "seed": seed,
                "source": "template_replacement",
                "route": "amide-replacement",
                "reaction_smarts": "[C:1](=[O:2])[N:3][*:4]>>[C:1](=[O:2])[N:3][*:5]",
                "reactants": [seed, block],
                "block": block,
                "canonical_smiles": candidate_smiles,
                "rank": idx,
            }
        )
    if not records:
        candidate_smiles = canonical_smiles("CCNCC(=O)Nc1ccccc1")
        records.append(
            {
                "candidate_id": candidate_id(candidate_smiles),
                "seed": seed,
                "source": "fallback",
                "route": "fallback",
                "reaction_smarts": "fallback",
                "reactants": [seed],
                "block": "CCN",
                "canonical_smiles": candidate_smiles,
                "rank": 0,
            }
        )
    return records


def _build_seed_analogs(seed: str, seed_mol: Chem.Mol, max_candidates: int) -> list[dict]:
    mutations: list[tuple[str, Callable[[Chem.RWMol], None]]] = []
    for atom in seed_mol.GetAtoms():
        if atom.GetSymbol() == "Cl" and atom.GetDegree() == 1:
            neighbor = atom.GetNeighbors()[0]
            if neighbor.GetIsAromatic():
                atom_idx = atom.GetIdx()
                for symbol, atomic_num in (("F", 9), ("Br", 35), ("I", 53)):
                    mutations.append(
                        (
                            f"aryl_Cl_to_{symbol}",
                            lambda rw_mol, idx=atom_idx, number=atomic_num: rw_mol.GetAtomWithIdx(idx).SetAtomicNum(number),
                        )
                    )
                mutations.append(
                    (
                        "aryl_Cl_to_H",
                        lambda rw_mol, idx=atom_idx: rw_mol.RemoveAtom(idx),
                    )
                )
                mutations.append(
                    (
                        "aryl_Cl_to_CF3",
                        lambda rw_mol, idx=atom_idx: _replace_chlorine_with_cf3(rw_mol, idx),
                    )
                )

    methoxy = Chem.MolFromSmarts("[OX2][CH3]")
    for match in seed_mol.GetSubstructMatches(methoxy):
        oxygen_idx, methyl_idx = match
        if not any(neighbor.GetIsAromatic() for neighbor in seed_mol.GetAtomWithIdx(oxygen_idx).GetNeighbors()):
            continue
        mutations.extend(
            [
                (
                    f"methoxy_to_ethoxy_at_{oxygen_idx}",
                    lambda rw_mol, idx=methyl_idx: _extend_methyl_to_ethyl(rw_mol, idx),
                ),
                (
                    f"methoxy_to_hydroxy_at_{oxygen_idx}",
                    lambda rw_mol, idx=methyl_idx: rw_mol.RemoveAtom(idx),
                ),
                (
                    f"methoxy_to_H_at_{oxygen_idx}",
                    lambda rw_mol, oxygen=oxygen_idx, methyl=methyl_idx: _remove_methoxy(
                        rw_mol, oxygen, methyl
                    ),
                ),
            ]
        )

    candidates: list[dict] = []
    seen_smiles = {seed}

    def add_mutant(
        mutation_names: list[str],
        actions: list[Callable[[Chem.RWMol], None]],
    ) -> None:
        editable = Chem.RWMol(seed_mol)
        try:
            for action in actions:
                action(editable)
            mutant = editable.GetMol()
            Chem.SanitizeMol(mutant)
            smiles = Chem.MolToSmiles(mutant, canonical=True)
        except (RuntimeError, ValueError):
            return
        if smiles in seen_smiles:
            return
        seen_smiles.add(smiles)
        candidates.append(
            {
                "candidate_id": candidate_id(smiles),
                "seed": seed,
                "source": "seed_analogue",
                "route": "+".join(mutation_names),
                "reaction_smarts": "",
                "reactants": [seed],
                "block": "",
                "canonical_smiles": smiles,
                "rank": len(candidates),
            }
        )

    for name, mutation in mutations:
        add_mutant([name], [mutation])
        if len(candidates) >= max_candidates:
            return candidates[:max_candidates]

    non_deletion_mutations = [
        (name, mutation)
        for name, mutation in mutations
        if (name.startswith("aryl_Cl_to_") and name != "aryl_Cl_to_H")
        or name.startswith("methoxy_to_ethoxy")
    ]
    for first_idx, (first_name, first_mutation) in enumerate(non_deletion_mutations):
        for second_name, second_mutation in non_deletion_mutations[first_idx + 1 :]:
            if first_name.startswith("aryl_Cl") == second_name.startswith("aryl_Cl"):
                continue
            add_mutant(
                [first_name, second_name],
                [first_mutation, second_mutation],
            )
            if len(candidates) >= max_candidates:
                return candidates[:max_candidates]
    return candidates[:max_candidates]


def _replace_chlorine_with_cf3(mol: Chem.RWMol, atom_idx: int) -> None:
    carbon = mol.GetAtomWithIdx(atom_idx)
    carbon.SetAtomicNum(6)
    carbon.SetIsAromatic(False)
    for _ in range(3):
        fluorine_idx = mol.AddAtom(Chem.Atom(9))
        mol.AddBond(atom_idx, fluorine_idx, Chem.BondType.SINGLE)


def _extend_methyl_to_ethyl(mol: Chem.RWMol, methyl_idx: int) -> None:
    carbon_idx = mol.AddAtom(Chem.Atom(6))
    mol.AddBond(methyl_idx, carbon_idx, Chem.BondType.SINGLE)


def _remove_methoxy(mol: Chem.RWMol, oxygen_idx: int, methyl_idx: int) -> None:
    for atom_idx in sorted((oxygen_idx, methyl_idx), reverse=True):
        mol.RemoveAtom(atom_idx)


def generate_candidates_for_seed(seed_smiles: str, max_candidates: int = 30) -> list[dict]:
    return build_replacement_candidates(seed_smiles, max_candidates=max_candidates)


def pairwise_similarity(smiles_a: str, smiles_b: str) -> float:
    if Chem is None:
        a_set = set(smiles_a)
        b_set = set(smiles_b)
        if not a_set and not b_set:
            return 1.0
        return len(a_set & b_set) / max(len(a_set | b_set), 1)
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)
