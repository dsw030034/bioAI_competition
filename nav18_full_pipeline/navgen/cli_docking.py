from __future__ import annotations

import argparse
import json

from .docking import DockingConfig, run_vina_docking


def main() -> int:
    parser = argparse.ArgumentParser(prog="navgen-docking")
    parser.add_argument("--config", required=True, help="Docking JSON configuration")
    args = parser.parse_args()
    result = run_vina_docking(DockingConfig.from_json(args.config))
    print(json.dumps({"n_ligands": len(result["results"]), "output": result["receptor_name"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
