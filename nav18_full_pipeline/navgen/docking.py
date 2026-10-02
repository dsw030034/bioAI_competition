from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DockingConfig:
    vina_executable: Path
    receptor_pdbqt: Path
    ligand_dir: Path
    output_dir: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    exhaustiveness: int = 8
    num_modes: int = 9
    cpu: int = 0
    seed: int = 0
    receptor_name: str = "nav18_site_c"
    receptor_pdb: Path | None = None
    ligand_sdf: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "DockingConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        required = ("vina_executable", "output_dir", "center", "size")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Docking config is missing required fields: {', '.join(missing)}")
        for key in ("center", "size"):
            if len(raw[key]) != 3:
                raise ValueError(f"Docking config field '{key}' must contain exactly three numbers")
        receptor_pdb = _resolve_optional_path(config_path.parent, raw.get("receptor_pdb"))
        receptor_pdbqt = _resolve_optional_path(config_path.parent, raw.get("receptor_pdbqt"))
        ligand_sdf = _resolve_optional_path(config_path.parent, raw.get("ligand_sdf"))
        ligand_dir = _resolve_optional_path(config_path.parent, raw.get("ligand_dir"))
        output_dir = _resolve_path(config_path.parent, raw["output_dir"])
        if receptor_pdbqt is None and receptor_pdb is not None:
            receptor_pdbqt = output_dir / "prepared_receptor.pdbqt"
        elif receptor_pdbqt is None:
            raise ValueError("Docking config requires either 'receptor_pdbqt' or 'receptor_pdb'.")
        if ligand_dir is None and ligand_sdf is not None:
            ligand_dir = output_dir / "ligands_pdbqt"
        elif ligand_dir is None:
            raise ValueError("Docking config requires either 'ligand_dir' or 'ligand_sdf'.")
        return cls(
            vina_executable=_resolve_path(config_path.parent, raw["vina_executable"]),
            receptor_pdbqt=receptor_pdbqt,
            ligand_dir=ligand_dir,
            output_dir=output_dir,
            center=tuple(float(value) for value in raw["center"]),
            size=tuple(float(value) for value in raw["size"]),
            exhaustiveness=int(raw.get("exhaustiveness", 8)),
            num_modes=int(raw.get("num_modes", 9)),
            cpu=int(raw.get("cpu", 0)),
            seed=int(raw.get("seed", 0)),
            receptor_name=str(raw.get("receptor_name", "nav18_site_c")),
            receptor_pdb=receptor_pdb,
            ligand_sdf=ligand_sdf,
        )


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_optional_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    return _resolve_path(base, value)


def _prepare_receptor_pdbqt(receptor_pdb: Path, output_path: Path, center: tuple[float, float, float], size: tuple[float, float, float]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executable = _find_vina_tool("mk_prepare_receptor", ["mk_prepare_receptor.exe", "mk_prepare_receptor"])
    command = [
        str(executable),
        "--read_pdb", str(receptor_pdb),
        "--write_pdbqt", str(output_path),
        "--default_altloc", "A",
        "--box_center", str(center[0]), str(center[1]), str(center[2]),
        "--box_size", str(size[0]), str(size[1]), str(size[2]),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to prepare receptor PDBQT with {executable}: {completed.stderr.strip() or completed.stdout.strip()}")
    return output_path


def _prepare_ligand_dir(ligand_sdf: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = _find_vina_tool("mk_prepare_ligand", ["mk_prepare_ligand.exe", "mk_prepare_ligand"])
    output_base = output_dir / f"{ligand_sdf.stem}.pdbqt"
    command = [str(executable), "-i", str(ligand_sdf), "-o", str(output_base)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to prepare ligand PDBQT with {executable}: {completed.stderr.strip() or completed.stdout.strip()}")
    prepared = sorted(output_dir.glob("*.pdbqt"))
    if not prepared:
        raise FileNotFoundError(f"No PDBQT outputs created from ligand SDF: {ligand_sdf}")
    return output_dir


def _find_vina_tool(tool_name: str, candidate_names: list[str]) -> str:
    preferred_envs = [
        Path.home() / "miniconda3/envs/vina_clean",
        Path.home() / "miniconda3/envs/vina",
    ]
    for name in candidate_names:
        local = Path(name)
        if local.exists() and local.is_file():
            return str(local)
        for env_root in preferred_envs:
            for subdir in ("Scripts", "bin", "Library/bin"):
                candidate = env_root / subdir / name
                if candidate.exists():
                    return str(candidate)
    search_paths = []
    for env_root in preferred_envs:
        search_paths.extend([
            env_root / "Scripts",
            env_root / "bin",
            env_root / "Library/bin",
        ])
    search_paths.append(Path.cwd())
    for base in search_paths:
        if base.exists():
            matches = list(base.glob(f"{tool_name}*")) + list(base.glob(f"{tool_name}.exe"))
            if matches:
                return str(matches[0])
    raise FileNotFoundError(
        f"Could not locate {tool_name}. Install the vina conda environment under "
        "the current user's Miniconda directory or provide a valid path in the config."
    )


def validate_inputs(config: DockingConfig) -> list[str]:
    errors: list[str] = []
    if not config.vina_executable.exists():
        errors.append(f"Vina executable not found: {config.vina_executable}")
    if not config.receptor_pdbqt.exists() and config.receptor_pdb is None:
        errors.append(f"receptor PDBQT not found: {config.receptor_pdbqt}")
    if config.receptor_pdb is not None and not config.receptor_pdb.exists():
        errors.append(f"receptor PDB not found: {config.receptor_pdb}")
    if not config.ligand_dir.exists() and config.ligand_sdf is None:
        errors.append(f"ligand directory not found: {config.ligand_dir}")
    if config.ligand_sdf is not None and not config.ligand_sdf.exists():
        errors.append(f"ligand SDF not found: {config.ligand_sdf}")
    if config.ligand_dir.exists() and not list(config.ligand_dir.glob("*.pdbqt")):
        errors.append(f"No ligand PDBQT files found in: {config.ligand_dir}")
    if any(value <= 0 for value in config.size):
        errors.append("Docking box sizes must be positive")
    if config.exhaustiveness < 1 or config.num_modes < 1:
        errors.append("exhaustiveness and num_modes must be positive")
    return errors


def _parse_best_affinity(stdout: str) -> float | None:
    for line in stdout.splitlines():
        match = re.match(r"\s*1\s+(-?\d+(?:\.\d+)?)\s+", line)
        if match:
            return float(match.group(1))
    return None


def prepare_docking_inputs(config: DockingConfig) -> DockingConfig:
    update = config
    if config.receptor_pdb is not None and (not config.receptor_pdbqt.exists() or not config.receptor_pdbqt.is_file()):
        prepared_receptor = _prepare_receptor_pdbqt(config.receptor_pdb, config.output_dir / "prepared_receptor.pdbqt", config.center, config.size)
        update = replace(config, receptor_pdbqt=prepared_receptor)
    if config.ligand_sdf is not None and (not config.ligand_dir.exists() or not list(config.ligand_dir.glob("*.pdbqt"))):
        prepared_dir = _prepare_ligand_dir(config.ligand_sdf, config.output_dir / "ligands_pdbqt")
        update = replace(update, ligand_dir=prepared_dir)
    errors = validate_inputs(update)
    if errors:
        raise FileNotFoundError("; ".join(errors))
    return update


def run_vina_docking(config: DockingConfig) -> dict[str, Any]:
    ready = prepare_docking_inputs(config)
    ready.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for ligand in sorted(ready.ligand_dir.glob("*.pdbqt")):
        output_pose = ready.output_dir / f"{ligand.stem}_out.pdbqt"
        command = [
            str(ready.vina_executable),
            "--receptor", str(ready.receptor_pdbqt),
            "--ligand", str(ligand),
            "--center_x", str(ready.center[0]),
            "--center_y", str(ready.center[1]),
            "--center_z", str(ready.center[2]),
            "--size_x", str(ready.size[0]),
            "--size_y", str(ready.size[1]),
            "--size_z", str(ready.size[2]),
            "--exhaustiveness", str(ready.exhaustiveness),
            "--num_modes", str(ready.num_modes),
            "--out", str(output_pose),
        ]
        if ready.cpu:
            command.extend(["--cpu", str(ready.cpu)])
        if ready.seed:
            command.extend(["--seed", str(ready.seed)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result = {
            "ligand": str(ligand),
            "output_pose": str(output_pose),
            "returncode": completed.returncode,
            "best_affinity_kcal_mol": _parse_best_affinity(completed.stdout),
            "stderr": completed.stderr.strip(),
        }
        if completed.returncode != 0:
            result["status"] = "failed"
        elif result["best_affinity_kcal_mol"] is None:
            result["status"] = "unparsed"
        else:
            result["status"] = "ok"
        results.append(result)
    summary = {
        "receptor_name": ready.receptor_name,
        "receptor_pdbqt": str(ready.receptor_pdbqt),
        "box": {"center": ready.center, "size": ready.size},
        "results": results,
        "warning": "Vina scores are docking proxies, not experimental binding affinities.",
    }
    (ready.output_dir / "docking_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
