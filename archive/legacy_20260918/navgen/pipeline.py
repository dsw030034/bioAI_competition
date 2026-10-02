from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import sqlite3

from rdkit import Chem, rdBase

from . import __version__
from .conformers import embed
from .evaluation import evaluate
from .generation import proposals, read_seeds
from .io import file_hash, write_csv, write_json
from .scoring import Scorer

EXPORT_FIELDS = ["candidate_id", "smiles", "reward", "similarity", "nearest_seed", "qed", "sa", "mw",
                 "logp", "tpsa", "heavy_atoms", "rotatable_bonds", "formal_charge", "membrane_proxy",
                 "selectivity_status", "ddg_pred", "ddg_std", "scaffold", "source", "parent_smiles", "route"]


def records(connection, passed=True):
    for (payload,) in connection.execute("SELECT payload FROM molecules WHERE passed=? ORDER BY reward DESC, candidate_id", (int(passed),)):
        yield json.loads(payload)


def run(config):
    seeds = read_seeds(config["seeds_file"])
    scorer = Scorer(config, seeds)
    out = Path(config["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {k: {"path": config[k], "sha256": file_hash(config[k])}
              for k in ("seeds_file", "building_blocks_file", "input_smiles", "reference_file") if config[k]}
    if config["selectivity"]["model"]:
        inputs["selectivity_model"] = {"path": config["selectivity"]["model"], "sha256": file_hash(config["selectivity"]["model"])}
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "version": __version__, "python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
                "config": config, "inputs": inputs, "seed_smiles": seeds,
                "limitations": ["2D membrane proxy is not permeability/logD or membrane free energy",
                                "Template route is not experimental synthesis validation",
                                "No target affinity or selectivity is inferred without a trained model"]}
    write_json(out / "manifest.json", manifest)
    conn = sqlite3.connect(out / "molecules.sqlite")
    try:
        conn.executescript("""
        CREATE TABLE molecules(candidate_id TEXT PRIMARY KEY, smiles TEXT UNIQUE, scaffold TEXT,
                               passed INTEGER, reward REAL, payload TEXT NOT NULL);
        CREATE TABLE attempts(attempt INTEGER PRIMARY KEY, raw_smiles TEXT, candidate_id TEXT,
                              status TEXT, source TEXT, provenance TEXT);
        CREATE TABLE conformers(candidate_id TEXT PRIMARY KEY, status TEXT, payload TEXT);
        CREATE INDEX rank_molecules ON molecules(passed, reward DESC);
        """)
        stats = {"attempts": 0, "valid": 0, "unique_valid": 0, "duplicates": 0,
                 "stop_reason": "generator_exhausted"}
        rejected = Counter()
        accepted = 0
        for proposal in itertools.islice(proposals(config, seeds), config["max_attempts"]):
            stats["attempts"] += 1
            result = scorer.score(proposal.smiles)
            status = "invalid"
            if result["valid"]:
                stats["valid"] += 1
                if conn.execute("SELECT 1 FROM molecules WHERE candidate_id=?", (result["candidate_id"],)).fetchone():
                    stats["duplicates"] += 1
                    status = "duplicate"
                else:
                    stats["unique_valid"] += 1
                    result.update(source=proposal.source, parent_smiles=proposal.parent_smiles, route=proposal.route)
                    conn.execute("INSERT INTO molecules VALUES (?,?,?,?,?,?)", (
                        result["candidate_id"], result["smiles"], result["scaffold"], int(result["passed"]),
                        result["reward"], json.dumps(result, ensure_ascii=False, allow_nan=False)))
                    status = "accepted" if result["passed"] else "rejected"
                    accepted += int(result["passed"])
            if status in {"invalid", "rejected"}:
                rejected.update(result["reasons"])
            conn.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?)", (
                stats["attempts"], proposal.smiles, result.get("candidate_id"), status, proposal.source,
                json.dumps({"parent_smiles": proposal.parent_smiles, "route": proposal.route, "reasons": result["reasons"]}, ensure_ascii=False)))
            if stats["attempts"] % 500 == 0:
                conn.commit()
            if accepted >= config["max_candidates"]:
                stats["stop_reason"] = "candidate_limit"
                break
        if stats["attempts"] == config["max_attempts"] and stats["stop_reason"] != "candidate_limit":
            stats["stop_reason"] = "attempt_limit"
        stats["rejection_counts"] = dict(rejected)
        conn.commit()
        write_csv(out / "candidates.csv", records(conn), EXPORT_FIELDS)
        with (out / "candidates.smi").open("w", encoding="utf-8") as handle:
            for record in records(conn):
                handle.write(f"{record['smiles']}\t{record['candidate_id']}\n")
        with (out / "candidates.jsonl").open("w", encoding="utf-8") as handle:
            for record in records(conn):
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        write_csv(out / "rejected.csv", records(conn, passed=False), ["candidate_id", "smiles", "reasons"])
        conf_options = config["conformers"]
        if conf_options["enabled"]:
            writer = Chem.SDWriter(str(out / "conformers.sdf"))
            best_writer = Chem.SDWriter(str(out / "best_conformers.sdf"))
            statuses = []
            try:
                for record in itertools.islice(records(conn), conf_options["max_molecules"]):
                    local_seed = (config["seed"] + int(record["candidate_id"][4:12], 16)) % (2**31 - 1)
                    mol, info = embed(record["smiles"], local_seed, conf_options["num_confs"], conf_options["max_iterations"])
                    info["candidate_id"] = record["candidate_id"]
                    statuses.append(info)
                    conn.execute("INSERT INTO conformers VALUES (?,?,?)", (record["candidate_id"], info["status"], json.dumps(info)))
                    if mol:
                        mol.SetProp("_Name", record["candidate_id"])
                        for key in ("candidate_id", "smiles"):
                            mol.SetProp(key, record[key])
                        mol.SetProp("forcefield", info["forcefield"])
                        for conf in info["conformers"]:
                            mol.SetIntProp("conformer_id", conf["conf_id"])
                            mol.SetDoubleProp("energy_kcal_mol", conf["energy"])
                            writer.write(mol, confId=conf["conf_id"])
                            if conf["conf_id"] == info["best_conf_id"]:
                                best_writer.write(mol, confId=conf["conf_id"])
            finally:
                writer.close()
                best_writer.close()
            write_json(out / "conformer_status.json", statuses)
            conn.commit()
        report = evaluate(conn, stats, config)
        write_json(out / "metrics.json", report)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        attempts=stats["attempts"], accepted=accepted)
        write_json(out / "manifest.json", manifest)
        return report
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(out / "manifest.json", manifest)
        raise
    finally:
        conn.close()

