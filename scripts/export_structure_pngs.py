"""Export existing 2D candidate libraries to paginated PNGs for Windows Photos."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw


def export(source, output):
    libraries = []
    for stem, label, prefix in (
        ("candidates", "主库", "main"),
        ("expanded_candidates", "扩展库", "expanded"),
        ("outside_candidates", "系列外保留", "outside"),
    ):
        if stem == "outside_candidates" and not (source / f"{stem}_2d.sdf").exists():
            continue
        mols = list(Chem.SDMolSupplier(str(source / f"{stem}_2d.sdf")))
        if any(mol is None for mol in mols):
            raise ValueError(f"Unreadable molecule in {stem}")
        libraries.append((mols, label, prefix))
    output.mkdir(parents=True, exist_ok=False)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 30) if font_path.exists() else ImageFont.load_default()
    index = []
    method_labels = {"site_single": "single", "site_pair": "pair", "site_triple": "triple",
                     "site_enumeration": "enumeration", "amide_cross_product": "cross",
                     "amide_recombination": "reaction", "mol2mol": "Mol2Mol"}
    for mols, label, prefix in libraries:
        pages = math.ceil(len(mols) / 12)
        print(f"Exporting {prefix}: {len(mols)} structures, {pages} pages", flush=True)
        for page in range(pages):
            batch = mols[page * 12:(page + 1) * 12]
            legends = []
            filename = f"{prefix}_{page + 1:02d}.png"
            for ordinal, mol in enumerate(batch, start=page * 12 + 1):
                cid = mol.GetProp("candidate_id")
                methods = json.loads(mol.GetProp("methods")) if mol.HasProp("methods") else []
                changes = f" | edits={mol.GetProp('changed_site_count')}" if mol.HasProp("changed_site_count") else ""
                legends.append(f"{ordinal:04d}  {cid}\n" + ", ".join(method_labels.get(m, m) for m in methods) + changes)
                index.append({"library": label, "number": ordinal, "candidate_id": cid,
                              "image": filename, "methods": ", ".join(methods)})
            options = Draw.MolDrawOptions()
            options.legendFontSize = 20
            options.legendFraction = 0.16
            options.bondLineWidth = 2
            grid = Draw.MolsToGridImage(batch, legends=legends, molsPerRow=3,
                                       subImgSize=(600, 420), drawOptions=options)
            canvas = Image.new("RGB", (grid.width, grid.height + 100), "white")
            canvas.paste(grid, (0, 100))
            ImageDraw.Draw(canvas).text((24, 24), f"A-803467 {label} | 第 {page + 1}/{pages} 页 | 共 {len(mols)} 个分子（编号非排名）",
                                        fill="black", font=font)
            canvas.save(output / filename)
    if (source / "seed_sites.png").is_file():
        shutil.copyfile(source / "seed_sites.png", output / "00_reference_sites.png")
    with (output / "index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["library", "number", "candidate_id", "image", "methods"])
        writer.writeheader()
        writer.writerows(index)
    (output / "查看说明.txt").write_text(
        "用 Windows 照片打开 PNG 图片，可用滚轮放大、方向键翻页。\n"
        "main 为主库，expanded 为扩展库，outside 为系列外保留集；如有 00_reference_sites 则为母体及位点。\n"
        "结构下方为序号、候选 ID 和来源：enumeration=位点枚举，single/pair/triple=单点/双点/三点，reaction=反应重组，Mol2Mol=模型采样。\n"
        "edits 为相对母体实际改变的允许位点数；同一结构可以有多种生成来源。\n"
        "编号是展示顺序，不是活性排名。index.csv 可按候选 ID 查找所在图片。\n", encoding="utf-8-sig")
    print(json.dumps({"candidates": len(index), "png_files": len(list(output.glob('*.png'))), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.input.resolve(), args.output.resolve())
