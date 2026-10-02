"""Repair missing heavy atoms, retain numbering, and prepare both rigid receptors."""
import argparse
import json
import re
import sys
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer
from openmm import Platform
from openmm.app import PDBFile
from navgen.docking_validation import select_chain, pdb_atoms, logged_run, sha256, write_json, pose_contacts

ROOT = Path(__file__).resolve().parents[1]


def prepare(out, resume=False):
    out.mkdir(parents=True, exist_ok=resume)
    for name, source, chain, ligand, resid in [
        ("nav18", "Nav1.8_structure/7WE4.pdb", "A", "95T", "2004"),
        ("nav15", "Nav1.5_structure/6LQA.pdb", "B", "QDN", "2110"),
    ]:
        directory = out / name
        directory.mkdir(exist_ok=resume)
        source = ROOT / source
        if resume and (directory / "box.json").is_file():
            audit = json.loads((directory / "audit.json").read_text())
            if audit["source_sha256"] != sha256(source) or audit["receptor_pdbqt_sha256"] != sha256(directory / "receptor.pdbqt"):
                raise ValueError("Resume inputs changed")
            print(f"Reusing verified {name}", flush=True)
            continue
        if name == "nav18":
            ref = Chem.SDMolSupplier(str(ROOT / "data/reference_7we4/a803467_bound.sdf"))[0]
            deposited = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                        for l in source.read_text().splitlines() if l.startswith("HETATM")
                        and l[17:20] == ligand and l[21] == chain and l[22:26].strip() == resid])
            if deposited.shape != ref.GetConformer().GetPositions().shape or not np.allclose(
                    deposited, ref.GetConformer().GetPositions(), atol=0.001):
                raise ValueError("Local receptor and deposited reference coordinates differ")
        else:
            # CCD provides molecular graph and stereochemistry; coordinates stay from 6LQA.
            url = "https://data.rcsb.org/rest/v1/core/chemcomp/QDN"
            raw = urllib.request.urlopen(url, timeout=60).read()
            (directory / "QDN_ccd.json").write_bytes(raw)
            ccd = json.loads(raw)
            smiles = ccd["rcsb_chem_comp_descriptor"]["SMILES_stereo"]
            template = Chem.MolFromSmiles(smiles)
            # RDKit's 3D perception does not retain neutral tertiary-amine N chirality.
            # Compare all carbon stereocentres; record this limitation for QDN.
            for atom in template.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0:
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            block = "\n".join(l for l in source.read_text().splitlines() if l.startswith("HETATM")
                    and l[17:20] == ligand and l[21] == chain and l[22:26].strip() == resid)
            ref = AllChem.AssignBondOrdersFromTemplate(template,
                    Chem.MolFromPDBBlock(block + "\nEND\n", sanitize=False, removeHs=True))
            Chem.AssignStereochemistryFrom3D(ref)
            if Chem.MolToSmiles(ref) != Chem.MolToSmiles(template):
                raise ValueError("QDN deposited stereochemistry differs from CCD")
        with Chem.SDWriter(str(directory / "reference.sdf")) as writer:
            writer.write(ref)
        ref_xyz = ref.GetConformer().GetPositions()
        selected = directory / "selected.pdb"
        select_chain(source, selected, chain)
        before = pdb_atoms(selected)
        platform = Platform.getPlatformByName("CPU")
        platform.setPropertyDefaultValue("Threads", "1")
        platform.setPropertyDefaultValue("DeterministicForces", "true")
        fixer = PDBFixer(filename=str(selected), platform=platform)
        fixer.findMissingResidues()
        fixer.missingResidues = {}  # Never invent unobserved loops or termini.
        fixer.findMissingAtoms()
        missing = [{"chain": r.chain.id, "residue": r.id, "name": r.name,
                    "missing_atoms": [a.name for a in aa],
                    "min_observed_distance_to_ligand_A": float(min(np.linalg.norm(p-ref_xyz, axis=1).min()
                        for k,p in before.items() if k[0] == r.chain.id and k[1] == r.id))}
                   for r, aa in fixer.missingAtoms.items()]
        fixer.addMissingAtoms(seed=20260930)
        repaired = directory / "repaired.pdb"
        with repaired.open("w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        after = pdb_atoms(repaired)
        if not set(before).issubset(after):
            raise ValueError("Repair removed atoms or changed residue identifiers")
        displacement = max(float(np.linalg.norm(after[k]-p)) for k,p in before.items())
        if displacement > 0.002:
            raise ValueError(f"Repair moved experimental atoms by {displacement} A")
        hetero = {}
        for l in source.read_text().splitlines():
            if l.startswith("HETATM"):
                key = f"{l[21]}:{l[22:27].strip()}:{l[17:20]}"
                p = np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                hetero[key] = min(hetero.get(key, float("inf")), float(np.linalg.norm(p-ref_xyz, axis=1).min()))
        audit = {"source": str(source), "source_sha256": sha256(source), "chain": chain,
                 "reference_ligand": f"{ligand}:{chain}:{resid}", "missing_heavy_atoms_repaired": missing,
                 "observed_heavy_atoms": len(before), "prepared_heavy_atoms": len(after),
                 "max_observed_atom_displacement_A": displacement, "missing_loops_built": False,
                 "numbering_preserved": True, "excluded_hetero_min_distance_A": hetero,
                 "repair_platform": "OpenMM CPU; Threads=1; DeterministicForces=true; seed=20260930",
                 "reference_sha256": sha256(directory / "reference.sdf"),
                 "reference_contacts": pose_contacts(ref, repaired),
                 "limitations": "Rigid protein-only baseline; native lipids/waters excluded; template protonation, no pKa enumeration."}
        if name == "nav15":
            audit["reference_chemistry_limitations"] = "CCD neutral QDN; carbon stereocentres verified; neutral tertiary N stereotag omitted by RDKit. Physiological protonation not validated."
        write_json(directory / "audit.json", audit)
        command = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor", "--read_pdb", repaired,
                    "-p", directory / "receptor.pdbqt", "--write_pdb", directory / "prepared.pdb",
                    "-j", directory / "receptor.json", "--default_altloc", "A"]
        exclusions = {}
        for attempt in range(4):
            log = directory / f"preparation_log_{attempt}.json"
            try:
                active_command = list(command)
                if exclusions:
                    filtered = directory / "explicitly_filtered.pdb"
                    filtered.write_text("\n".join(l for l in repaired.read_text().splitlines()
                        if not l.startswith("ATOM  ") or f"{l[21]}:{l[22:27].strip()}" not in exclusions) + "\n")
                    selected_filtered = directory / "meeko_input.pdb"
                    select_chain(filtered, selected_filtered, chain)
                    active_command[active_command.index("--read_pdb") + 1] = selected_filtered
                logged_run(active_command, log)
                break
            except RuntimeError:
                message = json.loads(log.read_text())["stderr"]
                bad = (set(re.findall(r"No template matched for residue_key='([^']+)'", message)) |
                       set(re.findall(r"corresponding to key ([A-Za-z0-9:]+)", message))) - set(exclusions)
                if not bad or attempt == 3:
                    raise
                for key in sorted(bad):
                    ch, residue = key.split(":")
                    coords = np.array([p for k,p in after.items() if k[0] == ch and k[1] == residue])
                    if not len(coords):
                        raise ValueError(f"Cannot audit {key}")
                    # Every atom must lie >8 A outside the search box (Vina interaction cutoff).
                    distance = float(np.linalg.norm(np.maximum(np.abs(coords-ref_xyz.mean(axis=0))-12, 0), axis=1).min())
                    if distance <= 8:
                        raise ValueError(f"Template failure near docking box at {key}; manual repair required")
                    exclusions[key] = {"min_distance_to_box_A": distance, "reason": "Meeko template/bond mismatch outside box +8A"}
                audit["explicit_distant_template_exclusions"] = exclusions
                write_json(directory / "audit.json", audit)
        audit["receptor_pdbqt_sha256"] = sha256(directory / "receptor.pdbqt")
        audit["repaired_pdb_sha256"] = sha256(repaired)
        write_json(directory / "audit.json", audit)
        write_json(directory / "box.json", {"center": ref_xyz.mean(axis=0).tolist(), "size": [24,24,24]})
        print(f"Prepared {name}: repaired {len(missing)} residues; retained numbering", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/docking_validation/receptors_v2")
    parser.add_argument("--resume", action="store_true", help="Reuse completed receptors only after hash checks")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.resume)
