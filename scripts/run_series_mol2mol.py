"""Portable, sampling-only Mol2Mol job. Run in an existing REINVENT4 environment."""
import argparse
import csv
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", type=Path, default=Path(__file__).with_name("a803467_generation.json"))
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("mol2mol_result"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    if a.samples < 1:
        p.error("samples must be positive")
    cfg = json.loads(a.series.read_text(encoding="utf-8"))
    missing = [name for name in ("reinvent", "torch", "rdkit") if util.find_spec(name) is None]
    if not a.prior.is_file():
        missing.append("prior_file")
    if not missing:
        import torch
        if a.device.startswith("cuda") and (not torch.cuda.is_available() or
                int(a.device.split(":")[1] if ":" in a.device else 0) >= torch.cuda.device_count()):
            missing.append("requested_cuda_device")
    report = {"python": sys.executable, "device": a.device, "missing": missing,
              "prior": str(a.prior.resolve()), "series_id": cfg["series_id"], "samples": a.samples}
    print(json.dumps(report, indent=2))
    if a.preflight:
        return 2 if missing else 0
    if missing:
        raise RuntimeError("Preflight failed: " + ", ".join(missing))
    from rdkit import Chem
    seed = Chem.MolToSmiles(Chem.MolFromSmiles(cfg["seed_smiles"]))
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seedpath = out / "seed.smi"
    seedpath.write_text(seed + "\n")
    # Only sampling configuration is written. No scoring, reward, or training stage.
    sampling = {"model_file": str(a.prior.resolve()), "smiles_file": str(seedpath),
                "sample_strategy": "multinomial", "temperature": 1.0,
                "output_file": str(out / "sampling_mol2mol.csv"), "num_smiles": a.samples,
                "unique_molecules": False, "randomize_smiles": True}
    value = lambda v: str(v).lower() if isinstance(v, bool) else json.dumps(v)
    toml = '\n'.join(['run_type = "sampling"', f'device = {json.dumps(a.device)}',
                       f'seed = {a.seed}', '', '[parameters]'] + [f'{k} = {value(v)}' for k,v in sampling.items()])
    (out / "sampling.toml").write_text(toml, encoding="utf-8")
    manifest = dict(report, seed_smiles=seed, prior_sha256=sha(a.prior), series_sha256=sha(a.series),
                    random_seed=a.seed, versions={}, status="started")
    for name in ("reinvent", "torch", "rdkit"):
        try:
            manifest["versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            manifest["versions"][name] = "source_checkout; version_not_installed"
    # Preserve the exact script/config used even when running from a copied remote bundle.
    (out / "series_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    manifest["runner_sha256"] = sha(__file__)
    command = [sys.executable, "-c", "from reinvent.Reinvent import main_script; main_script()", str(out / "sampling.toml")]
    manifest["command"] = command
    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    try:
        with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=out, timeout=7200)
        manifest["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError("REINVENT failed; inspect stderr.log and stdout.log")
        path = out / "sampling_mol2mol.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"SMILES", "Input_SMILES"}.issubset(reader.fieldnames or []):
                raise ValueError("Unexpected Mol2Mol CSV schema")
            rows = list(reader)
        if len(rows) != a.samples:
            raise ValueError(f"Expected {a.samples} raw samples, got {len(rows)}")
        if any(Chem.MolToSmiles(Chem.MolFromSmiles(r["Input_SMILES"])) != seed for r in rows):
            raise ValueError("Sample lineage does not match configured seed")
        manifest.update(status="completed", raw_rows=len(rows), output_sha256=sha(path))
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    print(f"Return the complete directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
