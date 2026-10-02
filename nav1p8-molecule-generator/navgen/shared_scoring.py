"""Identical seed constraints for offline candidates and RL batches."""
from rdkit import Chem
from .config import load_config
from .generation import load_seed_smiles, pairwise_similarity
from .scoring import score_candidate
from .site_rules import load_site_profile, score_site_c
from .protein_rules import ProteinWordScorer, load_protein_sequence


class ProjectScorer:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.seeds = load_seed_smiles(self.config["seed_file"])
        self.seed_mols = [Chem.MolFromSmiles(s) for s in self.seeds]
        path = self.config.get("site_profile_file")
        self.site = load_site_profile(path) if path else None
        path = self.config.get("protein_sequence_file")
        self.protein = ProteinWordScorer(load_protein_sequence(path)) if path else None

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles) if smiles.strip() else None
        if mol is None or mol.GetNumAtoms() == 0:
            return {"valid": False, "passed": False, "reward": 0.0, "reasons": ["invalid_smiles"]}
        if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return {"valid": True, "passed": False, "reward": 0.0, "reasons": ["disconnected_or_dummy"]}
        nearest = max(range(len(self.seeds)), key=lambda i: pairwise_similarity(smiles, self.seeds[i]))
        _, metrics = score_candidate(mol, self.seed_mols[nearest], self.config["filter_config"],
                                     self.protein.score(smiles) if self.protein else None,
                                     score_site_c(mol, self.site) if self.site else None)
        canonical = Chem.MolToSmiles(mol)
        if canonical in self.seeds:
            metrics["passed"] = False
            metrics["reasons"].append("seed_identity")
        if not metrics["passed"]:
            metrics["reward"] = 0.0
        return {**metrics, "valid": True, "nearest_seed": self.seeds[nearest]}
