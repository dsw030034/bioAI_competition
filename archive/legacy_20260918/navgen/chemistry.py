"""Canonicalization keeps stereochemistry, charge and tautomer identity explicit."""

from functools import lru_cache
import hashlib

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def molecule(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonical(smiles):
    mol = molecule(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def candidate_id(smiles):
    return "NAV_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]


def fingerprint(mol):
    return MORGAN.GetFingerprint(mol)


def scaffold(mol):
    # All acyclic molecules belong to one group in scaffold-based splitting.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


@lru_cache(maxsize=1)
def pains_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def descriptors(mol):
    return {
        "mw": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol), "qed": QED.qed(mol),
        "sa": float(sascorer.calculateScore(mol)),
        "heavy_atoms": mol.GetNumHeavyAtoms(), "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "pains": [entry.GetDescription() for entry in pains_catalog().GetMatches(mol)],
    }


def max_similarity(mol, reference_fps):
    return max(DataStructs.BulkTanimotoSimilarity(fingerprint(mol), reference_fps), default=0.0)

