"""Reference redocking gate, repeated sampling, and gated candidate screening.

Run from any directory with the project Python. Each invocation has isolated outputs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
from navgen.docking_validation import sha256, write_json, logged_run, fixed_frame_rmsd, pose_contacts

ROOT = Path(__file__).resolve().parents[1]


def prepare_ligand(smiles, seed, path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise ValueError("ETKDG embedding failed")
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError("Missing MMFF parameters")
    converged = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    if converged != 0:
        raise ValueError(f"MMFF failed to converge: {converged}")
    with Chem.SDWriter(str(path.with_suffix(".sdf"))) as w:
        w.write(mol)
    setups = MoleculePreparation().prepare(mol)
    if len(setups) != 1:
        raise ValueError("Unexpected multiple Meeko setups")
    text, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(error)
    path.write_text(text)


def one_job(job, receptors, out, vina, config):
    name, cid, conformer, seed, ligand, is_reference = job
    directory = out / f"{name}_{cid}_c{conformer}_s{seed}"
    directory.mkdir()
    recdir = receptors / name
    box = json.loads((recdir / "box.json").read_text())
    pose = directory / "poses.pdbqt"
    row = {"receptor": name, "candidate_id": cid, "conformer": conformer, "vina_seed": seed,
           "status": "failed", "is_reference": is_reference}
    try:
        command = [vina, "--receptor", recdir / "receptor.pdbqt", "--ligand", ligand,
                   "--out", pose, "--exhaustiveness", config["exhaustiveness"], "--num_modes", 9,
                   "--cpu", config["cpu_per_job"], "--seed", seed]
        for axis, center, size in zip("xyz", box["center"], box["size"]):
            command += [f"--center_{axis}", center, f"--size_{axis}", size]
        logged_run(command, directory / "vina_log.json", timeout=config["timeout_seconds"])
        if not pose.is_file() or not pose.stat().st_size:
            raise ValueError("Vina returned no pose")
        affinities = [float(x) for x in re.findall(r"REMARK VINA RESULT:\s+(-?[\d.]+)", pose.read_text())]
        if not affinities:
            raise ValueError("No affinity in pose output")
        pq = PDBQTMolecule(pose.read_text(), skip_typing=True)
        sdf, failures = RDKitMolCreate.write_sd_string(pq)
        if failures or not sdf:
            raise ValueError(f"Pose export failed: {failures}")
        sdfpath = directory / "poses.sdf"
        sdfpath.write_text(sdf)
        poses = list(Chem.SDMolSupplier(str(sdfpath), removeHs=True))
        if not poses or any(m is None for m in poses) or len(poses) != len(affinities):
            raise ValueError("Pose/score count mismatch")
        contacts = pose_contacts(poses[0], recdir / "repaired.pdb")
        row.update(status="ok", affinity_kcal_mol=affinities[0], all_pose_affinity_kcal_mol=affinities,
                   pose=str(sdfpath), **contacts)
        if is_reference:
            ref = Chem.SDMolSupplier(str(recdir / "reference.sdf"))[0]
            rmsds = [fixed_frame_rmsd(p, ref) for p in poses]
            reference_contacts = set(pose_contacts(ref, recdir / "repaired.pdb")["contact_residues_4A"])
            row.update(top1_rmsd_A=rmsds[0], best_of_modes_rmsd_A=min(rmsds), all_pose_rmsd_A=rmsds,
                       contact_recall=len(reference_contacts & set(contacts["contact_residues_4A"])) / max(1,len(reference_contacts)))
    except Exception as exc:
        row.update(status="failed", error=str(exc))
    write_json(directory / "result.json", row)
    return row


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["receptor"], r["candidate_id"]), []).append(r)
    summary = []
    for (name, cid), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        scores = [r["affinity_kcal_mol"] for r in ok]
        item = {"receptor": name, "candidate_id": cid, "successful": len(ok), "attempted": len(group)}
        if scores:
            item.update(median_affinity=float(np.median(scores)), min_affinity=min(scores),
                        max_affinity=max(scores), std_affinity=float(np.std(scores)),
                        clash_free_fraction=sum(r["heavy_pairs_below_2A"] == 0 for r in ok) / len(ok))
        if group[0]["is_reference"]:
            passes = [r for r in ok if r["top1_rmsd_A"] <= 2.0 and r["heavy_pairs_below_2A"] == 0
                      and r["contact_recall"] >= 0.5]
            item.update(top1_pass_fraction=len(passes)/len(group),
                        median_top1_rmsd_A=float(np.median([r["top1_rmsd_A"] for r in ok])) if ok else None,
                        gate_passed=len(ok) == len(group) and len(passes)/len(group) >= 0.8)
        summary.append(item)
    return summary


def run(config_path, stage, validation_report=None):
    config = json.loads(config_path.read_text())
    resolve = lambda s: (config_path.parent / s).resolve()
    receptors, vina = resolve(config["receptors"]), resolve(config["vina_executable"])
    receptor_names = config["receptor_names"]
    hashes = {name: {file: sha256(receptors / name / file) for file in
                    ("receptor.pdbqt", "reference.sdf", "repaired.pdb", "box.json")}
              for name in receptor_names}
    if stage == "candidates":
        if validation_report is None:
            raise ValueError("Candidate docking requires --validation-report")
        gate = json.loads(validation_report.read_text())
        if gate.get("stage") != "reference" or not gate.get("gate_passed"):
            raise ValueError("Reference validation did not pass; candidate ranking is blocked")
        if gate["input_hashes"] != hashes or gate["vina_sha256"] != sha256(vina):
            raise ValueError("Validated inputs changed; rerun reference validation")
        for key in ("exhaustiveness", "cpu_per_job"):
            if gate["config"][key] != config[key]:
                raise ValueError("Validated protocol differs from candidate protocol")
    out = resolve(config["output_root"]) / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + stage)
    out.mkdir(parents=True)
    nconf = config["reference_conformers"] if stage == "reference" else config["candidate_conformers"]
    if nconf < 1 or not config["vina_seeds"]:
        raise ValueError("Empty sampling budget")
    manifest = {"stage": stage, "config": config, "input_hashes": hashes, "vina_sha256": sha256(vina),
                "versions": {p: version(p) for p in ("rdkit", "meeko", "gemmi", "pdbfixer", "openmm")},
                "python": sys.executable, "validation_report": str(validation_report) if validation_report else None,
                "rmsd_method": "symmetry-corrected heavy atom RMSD in receptor frame; no ligand alignment",
                "gate_criteria": "all jobs successful; >=80% top1 RMSD <=2 A, zero heavy pairs <2 A, contact recall >=0.5",
                "limitations": "Rigid protein-only; one input protomer/tautomer; no calibrated subtype selectivity."}
    logged_run([vina, "--version"], out / "vina_version.json")
    entries = []
    if stage == "candidates":
        csvpath = resolve(config["candidate_csv"])
        manifest["candidate_csv_sha256"] = sha256(csvpath)
        with csvpath.open(encoding="utf-8", newline="") as f:
            entries = list(csv.DictReader(f))
        if len({r["candidate_id"] for r in entries}) != len(entries):
            raise ValueError("Duplicate candidate IDs")
        write_json(out / "candidate_inputs.json", entries)
    write_json(out / "manifest.json", manifest)
    jobs, rows = [], []
    prep = out / "ligands"
    prep.mkdir()
    for name in receptor_names:
        ref = Chem.SDMolSupplier(str(receptors / name / "reference.sdf"))[0]
        reference = {"candidate_id": "reference", "smiles": Chem.MolToSmiles(ref)}
        # Include native reference in each target's candidate batch; Nav1.5 also gets A803467.
        targets = [reference] + entries
        if stage == "candidates" and name != "nav18":
            a803 = Chem.SDMolSupplier(str(receptors / "nav18/reference.sdf"))[0]
            targets += [{"candidate_id": "a803467_control", "smiles": Chem.MolToSmiles(a803)}]
        for entry in targets:
            cid = entry["candidate_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", cid):
                raise ValueError("Unsafe candidate ID")
            for conformer in range(nconf):
                ligand = prep / f"{name}_{cid}_{conformer}.pdbqt"
                try:
                    prepare_ligand(entry["smiles"], config["embedding_seed"] + conformer, ligand)
                except Exception as exc:
                    for seed in config["vina_seeds"]:
                        rows.append({"receptor": name, "candidate_id": cid, "conformer": conformer,
                                     "vina_seed": seed, "status": "preparation_failed", "error": str(exc),
                                     "is_reference": cid == "reference"})
                    continue
                for seed in config["vina_seeds"]:
                    jobs.append((name, cid, conformer, seed, ligand, cid == "reference"))
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(one_job, j, receptors, out, vina, config) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(out / "results.json", rows)
            r = rows[-1]
            print(f"{len(rows)}/{len(jobs)} {r['receptor']} {r['candidate_id']} {r['status']} score={r.get('affinity_kcal_mol')} rmsd={r.get('top1_rmsd_A')}", flush=True)
    summary = summarize(rows)
    report = dict(manifest, summary=summary, gate_passed=stage == "reference" and
                  len(summary) == len(receptor_names) and all(s.get("gate_passed") for s in summary))
    write_json(out / "report.json", report)
    write_json(out / "results.json", rows)
    if summary:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in summary for k in r}))
            w.writeheader(); w.writerows(summary)
    print(f"Report: {out / 'report.json'}; reference gate={report['gate_passed']}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/docking_validation.json")
    parser.add_argument("--stage", choices=["reference", "candidates"], default="reference")
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args()
    run(args.config.resolve(), args.stage, args.validation_report)
