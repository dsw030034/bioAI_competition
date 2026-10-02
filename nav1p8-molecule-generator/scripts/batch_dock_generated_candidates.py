from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "docking_a803467_nav18_nav15.json"


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _subprocess_path(path: Path) -> str:
    return str(path.resolve())


def parse_affinity(stdout: str) -> float | None:
    for line in stdout.splitlines():
        match = re.match(r"\s*1\s+(-?\d+(?:\.\d+)?)\s+", line)
        if match:
            return float(match.group(1))
    return None


def _candidate_id(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid seed SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _read_inputs(config: dict[str, Any], config_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    candidate_csv = _resolve_path(config_dir, config["candidate_csv"])
    if not candidate_csv.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {candidate_csv}")
    with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "candidate_id" not in reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError(f"Candidates CSV must contain candidate_id and smiles: {candidate_csv}")
        fieldnames = list(reader.fieldnames)
        rows = []
        seen: dict[str, str] = {}
        for row in reader:
            candidate_id = str(row.get("candidate_id", "")).strip()
            smiles = str(row.get("smiles", "")).strip()
            if not candidate_id or not smiles:
                continue
            previous = seen.get(candidate_id)
            if previous is not None:
                if previous != smiles:
                    raise ValueError(f"Duplicate candidate_id {candidate_id} has conflicting SMILES")
                continue
            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError(f"Invalid candidate SMILES for {candidate_id}: {smiles}")
            seen[candidate_id] = smiles
            rows.append(row)

    seed_smiles = str(config.get("reference_seed_smiles", "")).strip()
    if seed_smiles:
        candidate_id = _candidate_id(seed_smiles)
        if candidate_id not in seen:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed_smiles,
                    "source": "seed_reference",
                    "route": "reference",
                    "smiles": Chem.MolToSmiles(Chem.MolFromSmiles(seed_smiles), canonical=True),
                }
            )
            fieldnames.extend(
                field for field in ("seed", "source", "route", "smiles") if field not in fieldnames
            )
    if not rows:
        raise ValueError(f"No valid candidate rows found in {candidate_csv}")
    return rows, fieldnames


def write_ligands(rows: list[dict[str, str]], raw_dir: Path) -> tuple[list[Path], Path]:
    sdf_paths = []
    for row in rows:
        candidate_id = row["candidate_id"]
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"Invalid SMILES for {candidate_id}: {row['smiles']}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(candidate_id[:8], 16) & 0x7FFFFFFF
        if AllChem.EmbedMolecule(mol, params) < 0:
            raise RuntimeError(f"RDKit could not generate a 3D conformer for {candidate_id}")
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
        else:
            raise RuntimeError(f"No MMFF or UFF parameters available for {candidate_id}")
        mol.SetProp("_Name", candidate_id)
        sdf_path = raw_dir / f"{candidate_id}.sdf"
        sdf_path.write_text(f"{Chem.MolToMolBlock(mol).rstrip()}\n$$$$\n", encoding="utf-8")
        sdf_paths.append(sdf_path)
    return sdf_paths, raw_dir.parent / "all_candidates.sdf"


def prepare_ligands(
    sdf_paths: list[Path],
    combined_sdf: Path,
    prepared_dir: Path,
    preparation_python: Path,
) -> None:
    combined_sdf.write_text(
        "".join(path.read_text(encoding="utf-8") for path in sdf_paths),
        encoding="utf-8",
    )
    command = [
        str(preparation_python),
        "-m",
        "meeko.cli.mk_prepare_ligand",
        "-i",
        _subprocess_path(combined_sdf),
        "--multimol_outdir",
        _subprocess_path(prepared_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Batch ligand preparation failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def dock_ligand(
    ligand_pdbqt: Path,
    receptor: dict[str, Any],
    receptor_pdbqt: Path,
    output_dir: Path,
    vina_executable: Path,
) -> tuple[float, str]:
    receptor_name = str(receptor["name"])
    pose_path = output_dir / "poses" / receptor_name / f"{ligand_pdbqt.stem}_out.pdbqt"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    center = receptor["center"]
    size = receptor["size"]
    command = [
        str(vina_executable),
        "--receptor",
        _subprocess_path(receptor_pdbqt),
        "--ligand",
        _subprocess_path(ligand_pdbqt),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(receptor.get("exhaustiveness", 8)),
        "--num_modes",
        str(receptor.get("num_modes", 9)),
        "--seed",
        str(receptor.get("seed", 20260923)),
        "--out",
        _subprocess_path(pose_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Vina failed for {receptor_name}/{ligand_pdbqt.stem}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    affinity = parse_affinity(completed.stdout)
    if affinity is None:
        raise RuntimeError(
            f"Could not parse Vina affinity for {receptor_name}/{ligand_pdbqt.stem}"
        )
    if not pose_path.is_file() or pose_path.stat().st_size == 0:
        raise RuntimeError(
            f"Vina produced no pose for {receptor_name}/{ligand_pdbqt.stem}"
        )
    return affinity, str(pose_path)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    rows, source_fields = _read_inputs(config, config_dir)
    vina_executable = _resolve_path(config_dir, config["vina_executable"])
    preparation_python = _resolve_path(config_dir, config["ligand_preparation_python"])
    output_root = _resolve_path(config_dir, config["output_root"])
    for label, path in (("Vina executable", vina_executable), ("preparation Python", preparation_python)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    receptors = config.get("receptors")
    if not isinstance(receptors, list) or len(receptors) < 2:
        raise ValueError("Config must define at least two receptors for subtype comparison")
    receptor_names: set[str] = set()
    receptor_paths: dict[str, Path] = {}
    for receptor in receptors:
        name = str(receptor.get("name", "")).strip()
        if not name or name in receptor_names:
            raise ValueError(f"Missing or duplicate receptor name: {name!r}")
        receptor_names.add(name)
        receptor_path = _resolve_path(config_dir, receptor["receptor_pdbqt"])
        if not receptor_path.is_file():
            raise FileNotFoundError(f"Receptor PDBQT not found for {name}: {receptor_path}")
        if len(receptor.get("center", [])) != 3 or len(receptor.get("size", [])) != 3:
            raise ValueError(f"Receptor {name} must have three-dimensional center and size")
        if any(float(value) <= 0 for value in receptor["size"]):
            raise ValueError(f"Receptor {name} has a non-positive docking box size")
        receptor_paths[name] = receptor_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / "docking_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raw_dir = output_dir / "ligands_sdf"
    prepared_dir = output_dir / "ligands_pdbqt"
    raw_dir.mkdir()
    prepared_dir.mkdir()

    for row in rows:
        for receptor in receptors:
            name = str(receptor["name"])
            row[f"{name}_best_affinity_kcal_mol"] = ""
            row[f"{name}_docking_status"] = "failed"
            row[f"{name}_docking_error"] = ""
            row[f"{name}_docking_pose"] = ""

    sdf_paths, combined_sdf = write_ligands(rows, raw_dir)
    try:
        prepare_ligands(sdf_paths, combined_sdf, prepared_dir, preparation_python)
    except Exception as exc:
        for row in rows:
            for receptor in receptors:
                row[f"{receptor['name']}_docking_error"] = str(exc)
    else:
        for row in rows:
            ligand_path = prepared_dir / f"{row['candidate_id']}.pdbqt"
            for receptor in receptors:
                name = str(receptor["name"])
                try:
                    if not ligand_path.is_file() or ligand_path.stat().st_size == 0:
                        raise RuntimeError(f"Meeko produced no PDBQT for {row['candidate_id']}")
                    affinity, pose_path = dock_ligand(
                        ligand_path,
                        receptor,
                        receptor_paths[name],
                        output_dir,
                        vina_executable,
                    )
                except Exception as exc:
                    row[f"{name}_docking_error"] = str(exc)
                    continue
                row[f"{name}_best_affinity_kcal_mol"] = str(affinity)
                row[f"{name}_docking_status"] = "ok"
                row[f"{name}_docking_pose"] = pose_path

    score_fields = [
        field
        for receptor in receptors
        for field in (
            f"{receptor['name']}_best_affinity_kcal_mol",
            f"{receptor['name']}_docking_status",
            f"{receptor['name']}_docking_error",
            f"{receptor['name']}_docking_pose",
        )
    ]
    output_csv = output_dir / "candidate_docking.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields + score_fields)
        writer.writeheader()
        writer.writerows(rows)
    for receptor in receptors:
        name = receptor["name"]
        succeeded = sum(row[f"{name}_docking_status"] == "ok" for row in rows)
        print(f"{name}: {succeeded}/{len(rows)} docked")
    print(f"Wrote {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Dock the same candidate set against multiple Nav subtypes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
