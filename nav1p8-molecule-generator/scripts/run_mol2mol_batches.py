"""Portable frozen-prior batches with verified resume and cumulative novelty counts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from rdkit import Chem, rdBase
from run_series_mol2mol import sha


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None and mol.GetNumAtoms() else None


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def collect(output, expected_seed, baseline, seeds, samples):
    combined, statistics, seen = [], [], set()
    for batch, seed in enumerate(seeds, 1):
        directory = output / f"batch_{batch:02d}"
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / "sampling_mol2mol.csv"
        if manifest["status"] != "completed" or manifest["output_sha256"] != sha(path):
            raise ValueError(f"Incomplete or modified batch: {directory}")
        if manifest["random_seed"] != seed or manifest["samples"] != samples:
            raise ValueError("Batch parameters differ from requested run")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != samples or any(canonical(r["Input_SMILES"]) != expected_seed for r in rows):
            raise ValueError("Batch row count or parent mismatch")
        own, invalid = set(), 0
        for index, row in enumerate(rows):
            smi = canonical(row["SMILES"])
            if smi:
                own.add(smi)
            else:
                invalid += 1
            combined.append({**row, "sampling_batch": batch, "sampling_random_seed": seed,
                             "sampling_row_in_batch": index, "sampling_batch_sha256": sha(path)})
        new = own - seen - {expected_seed}
        new_vs_all = new - baseline
        seen |= own
        statistics.append({"batch": batch, "random_seed": seed, "raw_rows": len(rows),
                           "invalid_rows": invalid, "batch_unique_including_seed": len(own),
                           "new_vs_previous_batches_excluding_seed": len(new),
                           "new_vs_previous_batches_and_baseline": len(new_vs_all),
                           "new_vs_previous_batches_and_baseline_fraction": len(new_vs_all) / len(rows),
                           "cumulative_unique_excluding_seed": len(seen - {expected_seed}),
                           "cumulative_new_vs_baseline": len(seen - baseline - {expected_seed})})
    fields = list(dict.fromkeys(k for row in combined for k in row))
    with (output / "sampling_combined.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    save(output / "batch_growth.json", statistics)
    with (output / "batch_growth.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    return {"raw_rows": len(combined), "combined_sha256": sha(output / "sampling_combined.csv"),
            "unique_excluding_seed": len(seen - {expected_seed}), "new_vs_baseline": len(seen - baseline - {expected_seed}),
            "batch_statistics": statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--samples-per-batch", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batches < 1 or args.samples_per_batch < 1:
        parser.error("Batch counts must be positive")
    runner = Path(__file__).with_name("run_series_mol2mol.py")
    expected_seed = canonical(json.loads(args.series.read_text())["seed_smiles"])
    if expected_seed is None:
        raise ValueError("Invalid configured seed")
    baseline = set()
    for line in args.baseline.read_text().splitlines():
        if line.strip():
            smi = canonical(line.split()[0])
            if smi is None:
                raise ValueError("Invalid baseline structure")
            baseline.add(smi)
    seeds = [args.first_seed + i for i in range(args.batches)]
    settings = {"series_sha256": sha(args.series), "prior_sha256": sha(args.prior),
                "runner_sha256": sha(runner), "batch_runner_sha256": sha(__file__),
                "baseline_sha256": sha(args.baseline), "baseline_unique": len(baseline),
                "seeds": seeds, "samples_per_batch": args.samples_per_batch, "device": args.device,
                "seed_smiles": expected_seed, "rdkit_version": rdBase.rdkitVersion,
                "ranking": "none", "policy": "fixed 5x200 default; preserve raw duplicates; no training or reward"}
    output = args.output.resolve()
    if output.exists():
        if not args.resume or json.loads((output / "settings.json").read_text()) != settings:
            raise ValueError("Output exists; resume requires identical input hashes and settings")
    else:
        output.mkdir(parents=True)
        save(output / "settings.json", settings)
    status = {"status": "running", "completed_batches": []}
    save(output / "manifest.json", status)
    try:
        for number, seed in enumerate(seeds, 1):
            directory = output / f"batch_{number:02d}"
            if directory.exists():
                m = json.loads((directory / "manifest.json").read_text())
                if (m["status"] != "completed" or m["prior_sha256"] != settings["prior_sha256"]
                        or m["series_sha256"] != settings["series_sha256"] or m["runner_sha256"] != settings["runner_sha256"]
                        or m["random_seed"] != seed or m["samples"] != args.samples_per_batch
                        or m["output_sha256"] != sha(directory / "sampling_mol2mol.csv")):
                    raise ValueError(f"Cannot reuse incomplete or changed batch {number}; use a new run directory")
            else:
                print(f"Starting batch {number}/{args.batches}, seed {seed}", flush=True)
                subprocess.run([sys.executable, str(runner), "--series", str(args.series.resolve()),
                                "--prior", str(args.prior.resolve()), "--device", args.device,
                                "--samples", str(args.samples_per_batch), "--seed", str(seed),
                                "--output", str(directory)], check=True)
            status["completed_batches"].append(number)
            save(output / "manifest.json", status)
        status.update(collect(output, expected_seed, baseline, seeds, args.samples_per_batch), status="completed")
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        save(output / "manifest.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
