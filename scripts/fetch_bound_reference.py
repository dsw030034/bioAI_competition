"""Fetch deposited 7WE4 ligand 95T A2004, assign seed bond orders, retain coordinates."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdkit import Chem
from rdkit.Chem import AllChem
from navgen.generation import load_seed_smiles


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "reference_7we4"
    if out.exists():
        raise FileExistsError(out)
    url = "https://files.rcsb.org/download/7WE4.pdb"
    raw = urllib.request.urlopen(url, timeout=40).read()
    lines = [l for l in raw.decode().splitlines() if l.startswith("HETATM") and
             l[17:20] == "95T" and l[21] == "A" and l[22:26].strip() == "2004"]
    seed = load_seed_smiles(root / "data/seeds_a803467.smi")[0]
    template = Chem.MolFromSmiles(seed)
    mol = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", sanitize=False, removeHs=False)
    if mol is None or mol.GetNumHeavyAtoms() != template.GetNumHeavyAtoms():
        raise ValueError("Missing/incomplete deposited ligand")
    original = mol.GetConformer().GetPositions().copy()
    mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    if Chem.MolToSmiles(mol) != seed:
        raise ValueError("Deposited ligand identity mismatch")
    if (mol.GetConformer().GetPositions() != original).any():
        raise ValueError("Coordinate preservation failed")
    out.mkdir()
    with Chem.SDWriter(str(out / "a803467_bound.sdf")) as writer:
        mol.SetProp("reference_source", "PDB 7WE4; 95T; auth chain A; auth residue 2004; deposited EM pose")
        writer.write(mol)
    (out / "source_ligand.pdb").write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    metadata = {"url": url, "source_pdb_sha256": hashlib.sha256(raw).hexdigest(),
                "smiles": seed, "ligand": "95T", "chain": "A", "residue": 2004,
                "method": "deposited EM coordinates; bond orders assigned from identity-checked seed",
                "reference_sdf_sha256": hashlib.sha256((out / "a803467_bound.sdf").read_bytes()).hexdigest()}
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
