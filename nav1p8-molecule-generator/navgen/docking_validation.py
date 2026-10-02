"""Auditable rigid-receptor docking and coordinate-frame-preserving validation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def logged_run(command, log_path, timeout=900):
    """Persist both streams, command and failures, including timeouts."""
    log_path = Path(log_path)
    try:
        result = subprocess.run([str(x) for x in command], capture_output=True,
                                text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        write_json(log_path, {"command": list(map(str, command)), "status": "timeout",
                             "stdout": str(exc.stdout or ""), "stderr": str(exc.stderr or "")})
        raise
    write_json(log_path, {"command": list(map(str, command)), "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log_path}: {result.stderr[-1200:]}")
    return result


def pdb_atoms(path):
    atoms = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM  ") and line[76:78].strip() not in ("H", "D"):
            key = (line[21], line[22:27].strip(), line[17:20], line[12:16].strip())
            atoms[key] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return atoms


def select_chain(source, destination, chain):
    """Keep incomplete residues and original identifiers; select a consistent altloc."""
    groups = {}
    for line in Path(source).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[21] != chain:
            continue
        if line[76:78].strip() in ("H", "D"):
            continue
        groups.setdefault(line[22:27], []).append(line)
    output = []
    previous_c = None
    for lines in groups.values():
        alternatives = {l[16] for l in lines if l[16] != " "}
        chosen = max(sorted(alternatives), key=lambda a: sum(float(l[54:60] or 0)
                     for l in lines if l[16] == a)) if alternatives else " "
        selected = {}
        for line in lines:
            if line[16] not in (" ", chosen):
                continue
            name = line[12:16].strip()
            if name not in selected or line[16] == " ":
                selected[name] = line[:16] + " " + line[17:]
        xyz = lambda l: np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        if output and (previous_c is None or "N" not in selected or
                       np.linalg.norm(previous_c - xyz(selected["N"])) > 2.0):
            output.append("TER")
        output.extend(selected.values())
        previous_c = xyz(selected["C"]) if "C" in selected else None
    if not output:
        raise ValueError(f"No ATOM records for chain {chain}")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("\n".join(output + ["TER", "END"]) + "\n")
    return len(groups), sum(l.startswith("ATOM  ") for l in output)


def fixed_frame_rmsd(pose, reference):
    """Symmetry-corrected heavy RMSD, WITHOUT aligning/translating the ligand."""
    pose, reference = Chem.RemoveHs(pose), Chem.RemoveHs(reference)
    if Chem.MolToSmiles(pose) != Chem.MolToSmiles(reference):
        raise ValueError("RMSD requires identical molecular graphs and stereochemistry")
    return float(rdMolAlign.CalcRMS(pose, reference, symmetrizeConjugatedTerminalGroups=True))


def pose_contacts(mol, receptor_pdb, cutoff=4.0):
    atoms = pdb_atoms(receptor_pdb)
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    distances = np.linalg.norm(xyz[:, None, :] - np.array(list(atoms.values()))[None, :, :], axis=2)
    contact_residues = sorted({":".join(k[:3]) for k, d in zip(atoms, distances.min(axis=0)) if d <= cutoff})
    return {"contact_residues_4A": contact_residues,
            "min_protein_heavy_distance_A": float(distances.min()),
            "heavy_pairs_below_2A": int((distances < 2.0).sum())}
