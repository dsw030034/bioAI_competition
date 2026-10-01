from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from clean_nav18_chain_a import RESIDUE_REQUIRED


def clean_chain_b_pdb(input_path: Path, output_path: Path) -> tuple[int, int]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from navgen.docking_validation import select_chain
    return select_chain(input_path, output_path, "B")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Nav1.5 chain B to preserved residues and original numbering."
    )
    parser.add_argument("--input", default="Nav1.5_structure/6LQA.pdb")
    parser.add_argument("--output", default="outputs/nav15_chainB_6LQA_clean.pdb")
    args = parser.parse_args()
    residue_count, atom_count = clean_chain_b_pdb(Path(args.input), Path(args.output))
    print(
        f"Wrote {args.output}: {residue_count} retained residues, "
        f"{atom_count} ATOM records."
    )


if __name__ == "__main__":
    main()
