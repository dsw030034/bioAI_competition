"""Verify the published snapshot and regenerate all 2D candidates on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_snapshot(snapshot):
    manifest = json.loads((snapshot / "sha256_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (snapshot / name).resolve()
        if snapshot.resolve() not in path.parents:
            raise ValueError("Snapshot manifest path leaves the snapshot directory")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Shared snapshot checksum mismatch: {name}")
    return len(manifest["files"])


def structures(path):
    pairs = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"Unexpected SMI schema: {path}")
    mapping = {cid: smi for smi, cid in pairs}
    if len(mapping) != len(pairs):
        raise ValueError(f"Duplicate structure IDs: {path}")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/shared_expansion_reproduced"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--full-gallery", action="store_true", help="Also export the complete paginated PNG library")
    args = parser.parse_args()
    snapshot = ROOT / "results/a803467_20261001"
    verified = verify_snapshot(snapshot)
    print(f"Verified {verified} published files", flush=True)
    if args.verify_only:
        return
    from navgen.generation_expansion import run
    summary = run(ROOT / "configs/a803467_expansion.json", snapshot / "baseline", args.output,
                  snapshot / "sampling/sampling_combined.csv")
    expected = json.loads((snapshot / "library/summary.json").read_text(encoding="utf-8"))
    for name in ("previous_unique_retained", "total_unique_retained", "new_unique_vs_baseline", "unique_by_scope"):
        if summary[name] != expected[name]:
            raise ValueError(f"Reproduced count differs from published results: {name}")
    for stem in ("candidates", "expanded_candidates", "outside_candidates"):
        if structures(args.output / f"{stem}.smi") != structures(snapshot / "library" / f"{stem}.smi"):
            raise ValueError(f"Reproduced structures differ: {stem}; check RDKit version")
    print("All 2,944 candidate IDs, SMILES and scope assignments match the published snapshot.", flush=True)
    if args.full_gallery:
        subprocess.run([sys.executable, str(ROOT / "scripts/export_structure_pngs.py"),
                        "--input", str(args.output.resolve()), "--output", str((args.output / "structure_png").resolve())], check=True)


if __name__ == "__main__":
    main()
