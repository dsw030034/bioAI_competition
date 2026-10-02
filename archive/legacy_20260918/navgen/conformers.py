"""ETKDG + MMFF/UFF structures; embedding/optimization failures stay missing."""

import math

from rdkit import Chem
from rdkit.Chem import AllChem

from .chemistry import molecule


def embed(smiles, seed, num_confs=5, max_iterations=300):
    mol = Chem.AddHs(molecule(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    try:
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
        if not ids:
            return None, {"status": "embedding_failed", "energy": None}
        if AllChem.MMFFHasAllMoleculeParams(mol):
            method = "MMFF94s"
            optimized = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=max_iterations, mmffVariant=method)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            method = "UFF"
            optimized = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=max_iterations)
        else:
            return None, {"status": "forcefield_unavailable", "energy": None}
        successful = [(conf_id, float(energy)) for conf_id, (status, energy) in zip(ids, optimized)
                      if status == 0 and math.isfinite(energy)]
        if not successful:
            return None, {"status": "optimization_failed", "energy": None, "n_embedded": len(ids)}
        best, energy = min(successful, key=lambda pair: pair[1])
        return mol, {"status": "ok", "energy": energy, "forcefield": method, "best_conf_id": best,
                     "n_embedded": len(ids), "n_converged": len(successful),
                     "conformers": [{"conf_id": i, "energy": e} for i, e in successful]}
    except (RuntimeError, ValueError) as error:
        return None, {"status": "embedding_or_optimization_error", "energy": None, "error": str(error)}

