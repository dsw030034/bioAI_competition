from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

RESIDUE_REQUIRED = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a Nav1.8 receptor to a chain-A only, standard-residue-only PDB")
    parser.add_argument("--input", default="Nav1.8_structure/7WE4.pdb", help="Input PDB file")
    parser.add_argument("--output", default="outputs/nav18_chainA_complete.pdb", help="Output PDB file")
    return parser.parse_args()


def is_standard_residue(name: str) -> bool:
    return name in RESIDUE_REQUIRED


def clean_chain_a_pdb(input_path: Path, output_path: Path) -> None:
    # This command selects a chain only; chemical repair is a separate audited step.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from navgen.docking_validation import select_chain
    residues, atoms = select_chain(input_path, output_path, "A")
    print(f"Wrote {output_path}: {residues} residues, {atoms} atoms; original IDs retained.")


if __name__ == '__main__':
    args = parse_args()
    clean_chain_a_pdb(Path(args.input), Path(args.output))
