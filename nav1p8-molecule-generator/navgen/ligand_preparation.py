"""Conservative input-state conformer pilot, with explicit lineage and failure records.

This is not a pH-state predictor, docking protocol, or activity filter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolAlign

from .series_generation import save_json, write_csv


STATE_HINTS = {
    "non_amide_amine_review": "[NX3;!$(N=*);!$(N#*);!$(NC=O);!$(NS(=O)=O);!$(NP(=O))]",
    "imine_or_amidine_review": "[NX2]=[CX3]",
    "aromatic_nitrogen_review": "[nH0;+0]",
    "phenol_review": "[OX2H]c",
    "carboxylic_acid_review": "[CX3](=O)[OX2H1]",
}


def state_record(candidate):
    mol = Chem.MolFromSmiles(candidate["smiles"])
    if mol is None or mol.GetNumAtoms() == 0:
        return None, {**candidate, "status": "invalid_smiles"}
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    digest = hashlib.sha256(smi.encode()).hexdigest()[:12]
    record = {**candidate, "input_smiles": candidate["smiles"], "state_smiles": smi,
              "state_id": candidate["candidate_id"] + ":input:" + digest,
              "formal_charge": Chem.GetFormalCharge(mol), "state_policy": "input_state_preserved",
              "protonation_status": "not_predicted", "tautomer_status": "not_enumerated",
              "state_review_hints": [name for name, smarts in STATE_HINTS.items() if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))],
              "status": "pending"}
    unspecified = [str(info.type) + ":" + str(info.centeredOn) for info in Chem.FindPotentialStereo(mol)
                   if info.specified != Chem.StereoSpecified.Specified]
    record["unspecified_stereo"] = unspecified
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        record["status"] = "requires_component_review"
    elif unspecified:
        # Do not make an arbitrary stereoisomer the prepared identity.
        record["status"] = "requires_stereochemistry_review"
    return mol, record


def geometry_issues(mol, conf_id, expected_smiles):
    xyz = np.asarray(mol.GetConformer(conf_id).GetPositions())
    if not np.isfinite(xyz).all():
        return ["nonfinite_coordinates"]
    issues = []
    table = Chem.GetPeriodicTable()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        radius_sum = sum(table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in (a, b))
        ratio = float(np.linalg.norm(xyz[a] - xyz[b])) / radius_sum
        if not 0.65 <= ratio <= 1.35:
            issues.append(f"bond_length:{a}-{b}")
    topology = Chem.GetDistanceMatrix(mol)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    for offset, a in enumerate(heavy):
        for b in heavy[offset+1:]:
            if topology[a, b] <= 2:
                continue
            radius_sum = sum(table.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in (a, b))
            if np.linalg.norm(xyz[a] - xyz[b]) < 0.60 * radius_sum:
                issues.append(f"heavy_atom_clash:{a}-{b}")
    check = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(check, confId=conf_id, replaceExistingTags=True)
    if Chem.MolToSmiles(Chem.RemoveHs(check), isomericSmiles=True) != expected_smiles:
        issues.append("stereochemistry_or_identity_mismatch")
    return issues


def prepare_candidate(candidate, num_confs=20, max_keep=5, random_seed=20261001):
    mol, state = state_record(candidate)
    if state["status"] != "pending":
        return state, [], None, []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = (random_seed + int(hashlib.sha256(state["state_id"].encode()).hexdigest()[:8], 16)) % (2**31 - 1)
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    params.enforceChirality = True
    state["embedding_seed"] = params.randomSeed
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    state["requested_conformers"], state["embedded_conformers"] = num_confs, len(ids)
    if not ids:
        state["status"] = "embedding_failed"
        return state, [], mol, []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        force_field = "MMFF94s"
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=force_field)
        factory = lambda idx: AllChem.MMFFGetMoleculeForceField(mol, props, confId=idx)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        force_field = "UFF"
        factory = lambda idx: AllChem.UFFGetMoleculeForceField(mol, confId=idx)
    else:
        state["status"] = "force_field_parameters_missing"
        return state, [], mol, []
    state["force_field"] = force_field
    records = []
    for idx in ids:
        record = {"candidate_id": candidate["candidate_id"], "state_id": state["state_id"],
                  "conformer_id": state["state_id"] + f":c{idx:03d}", "rdkit_conf_id": idx,
                  "force_field": force_field, "status": "pending", "geometry_issues": [],
                  "minimization_return_codes": [], "energy_kcal_mol": None, "relative_energy_same_state": None}
        try:
            ff = factory(idx)
            if ff is None:
                raise ValueError("Force field could not be constructed")
            code = ff.Minimize(maxIts=1000)
            record["minimization_return_codes"].append(code)
            if code == 1:
                code = ff.Minimize(maxIts=3000)
                record["minimization_return_codes"].append(code)
            energy = float(ff.CalcEnergy())
            if np.isfinite(energy):
                record["energy_kcal_mol"] = energy
            if code != 0:
                record["status"] = "optimization_not_converged"
            elif not np.isfinite(energy):
                record["status"] = "nonfinite_energy"
            else:
                record["geometry_issues"] = geometry_issues(mol, idx, state["state_smiles"])
                record["status"] = "geometry_failed" if record["geometry_issues"] else "geometry_passed"
        except (ValueError, RuntimeError) as exc:
            record["status"], record["error"] = "optimization_error", str(exc)
        records.append(record)
    usable = sorted((r for r in records if r["status"] == "geometry_passed"), key=lambda r: (r["energy_kcal_mol"], r["rdkit_conf_id"]))
    kept = []
    heavy = Chem.RemoveHs(mol)
    for record in usable:
        record["relative_energy_same_state"] = record["energy_kcal_mol"] - usable[0]["energy_kcal_mol"]
        duplicate = next((previous for previous in kept
                          if rdMolAlign.GetBestRMS(Chem.Mol(heavy), heavy, prbId=record["rdkit_conf_id"],
                                                  refId=previous["rdkit_conf_id"], maxMatches=10000) < 0.75), None)
        if duplicate:
            record["status"], record["duplicate_of"] = "redundant_geometry", duplicate["conformer_id"]
        elif len(kept) >= max_keep:
            record["status"] = "valid_over_output_cap"
        else:
            record["status"] = "retained"
            kept.append(record)
    state["retained_conformers"] = len(kept)
    state["status"] = "prepared_input_state" if kept else "no_usable_conformer"
    return state, records, mol, kept


def run(selection_path, output, num_confs=20, max_keep=5, random_seed=20261001):
    selection_path, output = Path(selection_path).resolve(), Path(output).resolve()
    if num_confs < 1 or max_keep < 1:
        raise ValueError("Conformer counts must be positive")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = selection["candidates"]
    if len({r["candidate_id"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs in pilot input")
    output.mkdir(parents=True, exist_ok=False)
    settings = {"selection_path": str(selection_path), "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                "rdkit_version": rdBase.rdkitVersion, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "num_confs": num_confs, "max_keep": max_keep, "random_seed": random_seed,
                "state_policy": "preserve input charge, tautomer and specified stereo; no salt stripping or neutralization",
                "unspecified_stereo_policy": "flag for review; no arbitrary stereoisomer embedding",
                "embedding": "ETKDGv3; single thread; preoptimization RMS pruning 0.5 A",
                "optimization": "MMFF94s if complete parameters, else UFF; 1000 then 3000 iterations; convergence required",
                "geometry_checks": "finite coordinates; covalent-radius bond ratio 0.65..1.35; heavy nonbonded distance >=0.60*vdW sum for graph distance>2; stereo identity",
                "deduplication": "symmetry-aware aligned heavy atom RMSD <0.75 A within same state; then output cap",
                "energy_comparison": "within the same chemical state and force field only; not affinity or cross-molecule ranking",
                "preparation_scope": "3D feasibility pilot only; pH-state coverage and docking readiness not established"}
    save_json(output / "settings.json", settings)
    save_json(output / "input_snapshot.json", selection)
    states, records = [], []
    with Chem.SDWriter(str(output / "prepared_input_states_3d.sdf")) as writer:
        for candidate in candidates:
            print(f"Preparing {candidate['candidate_id']} ({candidate['scope']})", flush=True)
            try:
                state, confs, mol, kept = prepare_candidate(candidate, num_confs, max_keep, random_seed)
                # Export the actual confId, not the molecule's default/first conformer.
                for conformer in kept:
                    exported = Chem.Mol(mol)
                    for key, value in {**state, **conformer}.items():
                        exported.SetProp(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
                    writer.write(exported, confId=conformer["rdkit_conf_id"])
            except (ValueError, RuntimeError) as exc:
                state = {**candidate, "status": "preparation_error", "error": str(exc)}
                confs = []
            states.append(state)
            records.extend(confs)
    summary = {"candidates": len(states), "states": dict(Counter(r["status"] for r in states)),
               "conformers": dict(Counter(r["status"] for r in records)),
               "retained_conformers": sum(r["status"] == "retained" for r in records),
               "requires_state_review_hint": sum(bool(r.get("state_review_hints")) for r in states),
               "protonation_and_tautomer_coverage": "not assessed", "binding_and_activity": "not assessed"}
    save_json(output / "states.json", states)
    save_json(output / "conformers.json", records)
    save_json(output / "summary.json", summary)
    write_csv(output / "state_status.csv", states)
    write_csv(output / "conformer_status.csv", records)
    (output / "查看说明.txt").write_text(
        "本目录为输入化学状态下的三维准备试跑。SDF 为真实 3D 坐标；不含受体/对接姿势。\n"
        "candidate_id → state_id → conformer_id 保留来源。所有尝试、失败和去重结果均见状态文件。\n"
        "不自动去盐、中和、预测 pH 状态、枚举互变异构体或猜测未定立体化学；相关标记待审阅。\n"
        "能量仅用于同一状态内部选择构象，不能比较不同分子的活性。几何通过不代表结合或可合成。\n"
        "这是可复现的构象生成试跑，尚不是已完成质子化审查的 docking 输入库。\n", encoding="utf-8-sig")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-confs", type=int, default=20)
    parser.add_argument("--max-keep", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.selection, args.output, args.num_confs, args.max_keep), ensure_ascii=False, indent=2))
