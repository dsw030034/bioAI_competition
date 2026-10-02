"""Template-based local analog enumeration plus CLM-output ingestion.

Amide substitutions preserve one seed-side fragment. Routes are hypotheses,
not experimentally validated synthesis or evidence of reagent availability.
"""

from dataclasses import dataclass, field
import csv
import random

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from .chemistry import canonical, molecule
from .io import read_smiles

AMIDE_FORWARD = "[C:1](=[O:2])[O;H1].[N;H1,H2;!$(N-C=O):3]>>[C:1](=[O:2])[N:3]"
AMIDE_RETRO = "[C:1](=[O:2])[N;H1,H2:3]>>[C:1](=[O:2])O.[N:3]"


@dataclass
class Proposal:
    smiles: str
    source: str
    parent_smiles: str | None = None
    route: dict = field(default_factory=dict)


def read_seeds(path):
    seeds = []
    for raw in read_smiles(path):
        smi = canonical(raw)
        mol = molecule(smi) if smi else None
        if mol is None or len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            raise ValueError(f"Invalid or disconnected seed: {raw}")
        if smi not in seeds:
            seeds.append(smi)
    if not seeds:
        raise ValueError("Seed file is empty")
    return seeds


def read_blocks(path):
    blocks = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "role", "smiles"} <= set(reader.fieldnames or []):
            raise ValueError("Building blocks require id,role,smiles columns")
        ids = set()
        queries = {"acid": Chem.MolFromSmarts("C(=O)[O;H1]"),
                   "amine": Chem.MolFromSmarts("[N;H1,H2;!$(N-C=O)]")}
        for row in reader:
            mol = molecule(row["smiles"])
            if row["id"] in ids or row["role"] not in queries or mol is None:
                raise ValueError(f"Invalid or duplicate building block: {row}")
            if len(Chem.GetMolFrags(mol)) != 1 or not mol.HasSubstructMatch(queries[row["role"]]):
                raise ValueError(f"Building block does not match declared role: {row['id']}")
            ids.add(row["id"])
            blocks.append((row["id"], row["role"], mol))
    if not blocks:
        raise ValueError("Building block file is empty")
    return blocks


def reaction_proposals(parents, blocks, seed):
    retro = rdChemReactions.ReactionFromSmarts(AMIDE_RETRO)
    forward = rdChemReactions.ReactionFromSmarts(AMIDE_FORWARD)
    blocks = list(blocks)
    random.Random(seed).shuffle(blocks)
    for parent in parents:
        mol = molecule(parent)
        if mol is None:
            continue
        for acid, amine in retro.RunReactants((mol,), maxProducts=100):
            try:
                Chem.SanitizeMol(acid)
                Chem.SanitizeMol(amine)
            except (ValueError, RuntimeError):
                continue
            # Alternate both sides instead of consuming the entire budget on one side.
            for block_id, role, block in blocks:
                reactants = (block, amine) if role == "acid" else (acid, block)
                for products in forward.RunReactants(reactants, maxProducts=100):
                    product = products[0]
                    try:
                        Chem.SanitizeMol(product)
                    except (ValueError, RuntimeError):
                        continue
                    smi = canonical(Chem.MolToSmiles(product))
                    if smi:
                        yield Proposal(smi, "amide_reaction", parent, {
                            "template": "amide_coupling", "reaction_smarts": AMIDE_FORWARD,
                            "reactants": [canonical(Chem.MolToSmiles(m)) for m in reactants],
                            "building_block_id": block_id, "replaced_side": role,
                            "validation": "template_only", "availability_verified": False,
                        })


def proposals(config, seeds):
    backend = config["backend"]
    if backend == "reaction":
        yield from reaction_proposals(seeds, read_blocks(config["building_blocks_file"]), config["seed"])
    elif backend == "smiles":
        for smi in read_smiles(config["input_smiles"]):
            yield Proposal(smi, "clm_import")
    else:
        blocks = read_blocks(config["building_blocks_file"])
        yield from reaction_proposals(seeds, blocks, config["seed"])
        for i, smi in enumerate(read_smiles(config["input_smiles"])):
            yield Proposal(smi, "clm_import")
            yield from reaction_proposals([smi], blocks, (config["seed"] + i) % (2**31))
