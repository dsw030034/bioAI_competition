"""Make a small portable Mol2Mol job bundle without model weights or credentials."""
import argparse
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or out.with_suffix(".zip").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/a803467_generation.json", out / "a803467_generation.json")
    shutil.copyfile(ROOT / "scripts/run_series_mol2mol.py", out / "run_series_mol2mol.py")
    shutil.copyfile(ROOT / "docs/MOL2MOL_REMOTE.md", out / "README.md")
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):
            z.write(file, arcname=out.name + "/" + file.name)
    print(out.with_suffix(".zip"))


if __name__ == "__main__":
    main()
