import math

from rdkit import Chem, DataStructs

from .chemistry import canonical, candidate_id, descriptors, fingerprint, molecule, scaffold


def interval_score(value, low, high):
    """Plateau inside a preferred interval, symmetric Gaussian tails outside."""
    distance = max(low - value, 0.0, value - high)
    return math.exp(-0.5 * (distance / ((high - low) / 2)) ** 2)


class Scorer:
    def __init__(self, config, seeds):
        self.config = config
        self.seeds = seeds
        self.seed_fps = [fingerprint(molecule(s)) for s in seeds]
        query = config["filters"]["required_smarts"]
        self.required = Chem.MolFromSmarts(query) if query else None
        if query and self.required is None:
            raise ValueError("Invalid required_smarts")
        self.selectivity = None
        if config["selectivity"]["model"]:
            from .selectivity import SelectivityModel
            self.selectivity = SelectivityModel(config["selectivity"]["model"], config)

    def score(self, raw):
        smi = canonical(raw)
        if smi is None:
            return {"raw_smiles": raw, "smiles": None, "valid": False,
                    "passed": False, "reasons": ["invalid_smiles"], "reward": 0.0}
        mol = molecule(smi)
        result = {"candidate_id": candidate_id(smi), "smiles": smi, "valid": True,
                  "scaffold": scaffold(mol), "reasons": []}
        f = self.config["filters"]
        reasons = result["reasons"]
        if len(Chem.GetMolFrags(mol)) != 1:
            reasons.append("disconnected_structure")
        if any(a.GetAtomicNum() not in {1, 6, 7, 8, 9, 15, 16, 17, 35, 53} for a in mol.GetAtoms()):
            reasons.append("unsupported_element")
        if reasons:
            return {**result, "passed": False, "reward": 0.0}
        result.update(descriptors(mol))
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint(mol), self.seed_fps)
        nearest = max(range(len(similarities)), key=similarities.__getitem__)
        result.update(similarity=float(similarities[nearest]), nearest_seed=self.seeds[nearest])
        if f["exclude_seeds"] and smi in self.seeds:
            reasons.append("seed_identity")
        for name in ("mw", "logp", "tpsa"):
            if not f[name][0] <= result[name] <= f[name][1]:
                reasons.append(name)
        for name in ("qed", "similarity"):
            if result[name] < f[name + "_min"]:
                reasons.append(name)
        for name in ("sa", "heavy_atoms", "rotatable_bonds"):
            if result[name] > f[name + "_max"]:
                reasons.append(name)
        if f["pains"] and result["pains"]:
            reasons.append("pains")
        if self.required is not None and not mol.HasSubstructMatch(self.required):
            reasons.append("required_substructure")
        membrane = math.prod(interval_score(result[k], *v) for k, v in self.config["membrane"].items()) ** (1 / 3)
        scores = {"similarity": result["similarity"], "qed": result["qed"],
                  "sa": max(0.0, min(1.0, (10 - result["sa"]) / 9)), "membrane": membrane,
                  "size": min(1.0, self.config["membrane"]["mw"][1] / result["mw"])}
        result["selectivity_status"] = "not_available"
        result["ddg_pred"] = None
        result["ddg_std"] = None
        if self.selectivity:
            prediction = self.selectivity.predict(smi)
            result.update(prediction)
            s = self.config["selectivity"]
            in_domain = prediction["training_similarity"] >= s["ood_similarity_min"]
            result["selectivity_status"] = "in_domain" if in_domain else "out_of_domain"
            if not in_domain and self.config["weights"]["selectivity"] > 0:
                reasons.append("selectivity_out_of_domain")
            lower = prediction["ddg_pred"] - s["uncertainty_penalty"] * prediction["ddg_std"]
            z = max(-60, min(60, lower / s["reward_scale"]))
            scores["selectivity"] = 1 / (1 + math.exp(-z))
        weights = self.config["weights"]
        reward = sum(weights[k] * v for k, v in scores.items()) / sum(weights[k] for k in scores)
        result.update(components=scores, passed=not reasons, reward=reward if not reasons else 0.0,
                      membrane_proxy=membrane, membrane_method="2d_descriptor_intervals")
        return result

