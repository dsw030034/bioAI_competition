from __future__ import annotations

import json
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - fallback for minimal environments
    Chem = None
    AllChem = None


def generate_conformers(mol, num_confs: int = 5) -> tuple[bool, list[dict]]:
    if Chem is None:
        return True, [{"conf_id": 0, "status": "fallback", "energy": 0.0}]
    if mol is None:
        return False, [{"status": "invalid_molecule"}]

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, clearConfs=True)
    if not conf_ids:
        return False, [{"status": "embedding_failed"}]

    records: list[dict] = []
    for conf_id in conf_ids:
        props = {}
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
            if ff is not None:
                ff.Minimize(maxIts=200)
                energy = ff.CalcEnergy()
                props["energy"] = float(energy)
            else:
                props["energy"] = None
        except Exception:
            try:
                uff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                if uff is not None:
                    uff.Initialize()
                    uff.Minimize(maxIts=200)
                    props["energy"] = float(uff.CalcEnergy())
                else:
                    props["energy"] = None
            except Exception:
                props["energy"] = None
        records.append({"conf_id": int(conf_id), "status": "ok", **props})
    return True, records


def export_conformers(candidates: list[dict], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / "conformers.sdf"
    best_path = output_dir / "best_conformers.sdf"
    status_path = output_dir / "conformer_status.json"

    if Chem is None:
        status_path.write_text(json.dumps({"status": "rdkit_unavailable_fallback"}, indent=2), encoding="utf-8")
        return {"sdf": str(sdf_path), "best_sdf": str(best_path), "status": str(status_path)}

    writer = Chem.SDWriter(str(sdf_path))
    best_writer = Chem.SDWriter(str(best_path))
    status: dict[str, str] = {}
    for record in candidates:
        smiles = record.get("canonical_smiles") or record.get("smiles")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            status[record["candidate_id"]] = "invalid_smiles"
            continue
        ok, confs = generate_conformers(mol, num_confs=max(1, record.get("num_confs", 5)))
        if not ok:
            status[record["candidate_id"]] = "embedding_failed"
            continue

        best_energy = min((c.get("energy") for c in confs if c.get("energy") is not None), default=None)
        if best_energy is None:
            status[record["candidate_id"]] = "optimization_failed"
            continue

        conf_status = f"ok:{len(confs)}"
        status[record["candidate_id"]] = conf_status
        for conf in confs:
            conf_mol = Chem.Mol(mol)
            conf_mol.SetProp("candidate_id", record["candidate_id"])
            conf_mol.SetProp("conformer_id", str(conf["conf_id"]))
            if conf.get("energy") is not None:
                conf_mol.SetProp("energy", str(conf["energy"]))
            writer.write(conf_mol)
        best_conf = min(confs, key=lambda item: item.get("energy") if item.get("energy") is not None else 1e18)
        best_mol = Chem.Mol(mol)
        best_mol.SetProp("candidate_id", record["candidate_id"])
        best_mol.SetProp("conformer_id", str(best_conf["conf_id"]))
        if best_conf.get("energy") is not None:
            best_mol.SetProp("energy", str(best_conf["energy"]))
        best_writer.write(best_mol)

    writer.close()
    best_writer.close()
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return {"sdf": str(sdf_path), "best_sdf": str(best_path), "status": str(status_path)}
