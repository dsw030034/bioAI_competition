"""Create a separate +1 quinidine reference control, preserving heavy coordinates."""
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rdkit import Chem
from navgen.docking_validation import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "outputs/docking_validation/receptors_v2/nav15"
    target = source.parent / "nav15_qdn_protonated"
    mol = Chem.SDMolSupplier(str(source / "reference.sdf"))[0]
    xyz = mol.GetConformer().GetPositions().copy()
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and not a.GetIsAromatic()
                 and a.GetDegree() == 3 and a.GetFormalCharge() == 0]
    if len(nitrogens) != 1:
        raise ValueError("Expected one neutral tertiary aliphatic amine")
    n = nitrogens[0]
    n.SetFormalCharge(1)
    n.SetNumExplicitHs(1)
    n.SetNoImplicit(True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    if not np.array_equal(xyz, mol.GetConformer().GetPositions()):
        raise ValueError("Protonation moved reference heavy atoms")
    target.mkdir(exist_ok=False)
    for filename in ("receptor.pdbqt", "repaired.pdb", "prepared.pdb", "box.json", "receptor.json"):
        shutil.copyfile(source / filename, target / filename)
    with Chem.SDWriter(str(target / "reference.sdf")) as w:
        w.write(mol)
    audit = json.loads((source / "audit.json").read_text())
    audit.update(reference_charge=1, reference_smiles=Chem.MolToSmiles(mol),
                 reference_sha256=sha256(target / "reference.sdf"),
                 reference_chemistry_limitations="Tertiary aliphatic amine protonated; no pKa calculation. A separate charge-state sensitivity control, not a newly observed experimental ligand.")
    write_json(target / "audit.json", audit)
    print(target)


if __name__ == "__main__":
    main()
