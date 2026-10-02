"""Generation-only series workflow: explicit sites, complete lineage, no reward ranking."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem, Draw
from .generation import candidate_id, canonical_smiles
from .reaction_library import enumerate_library


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Series:
    def __init__(self, config):
        self.config = config
        self.seed = canonical_smiles(config["seed_smiles"])
        scaffold = Chem.MolFromSmiles(config["attachment_scaffold"])
        if scaffold is None:
            raise ValueError("Invalid attachment scaffold")
        labels = [a.GetAtomMapNum() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 0]
        if len(set(labels)) != len(labels) or set(labels) != {s["label"] for s in config["sites"]}:
            raise ValueError("Attachment labels must be unique and match sites")
        rw = Chem.RWMol(scaffold)
        anchors = {}
        for a in scaffold.GetAtoms():
            if a.GetAtomicNum() == 0:
                if a.GetDegree() != 1:
                    raise ValueError("Each dummy must have one neighbor")
                anchor = a.GetNeighbors()[0].GetIdx()
                rw.GetAtomWithIdx(anchor).SetAtomMapNum(100 + a.GetAtomMapNum())
        for idx in sorted([a.GetIdx() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
            rw.RemoveAtom(idx)
        self.core = rw.GetMol()
        Chem.SanitizeMol(self.core)
        for a in self.core.GetAtoms():
            if a.GetAtomMapNum():
                anchors[a.GetAtomMapNum() - 100] = a.GetIdx()
                a.SetAtomMapNum(0)
        self.anchors = anchors
        # Query preserves ring connectivity; boundary checks below restrict ALL new bonds to named sites.
        self.query = Chem.MolFromSmarts(Chem.MolToSmarts(self.core))
        self.relaxed = Chem.MolFromSmarts(config["round2_core_smarts"])
        self.original = {s["label"]: s["original"] for s in config["sites"]}
        if self.assemble(self.original) != self.seed:
            raise ValueError("Attachment scaffold does not reconstruct the configured seed")
        for s in config["sites"]:
            if s["original"] not in s["fragments"]:
                raise ValueError("Original fragment missing from site vocabulary")
            for fragment in s["fragments"]:
                if fragment:
                    m = Chem.MolFromSmiles(fragment)
                    if m is None or len(Chem.GetMolFrags(m)) != 1 or any(a.GetAtomicNum() == 0 for a in m.GetAtoms()):
                        raise ValueError(f"Invalid single-component fragment {fragment}")

    def assemble(self, fragments):
        rw = Chem.RWMol(self.core)
        for label, fragment in fragments.items():
            if not fragment:
                continue  # empty fragment represents H at the original site
            frag = Chem.MolFromSmiles(fragment)
            offset = rw.GetNumAtoms()
            rw = Chem.RWMol(Chem.CombineMols(rw.GetMol(), frag))
            rw.AddBond(self.anchors[label], offset, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    def strict_match(self, mol):
        for match in mol.GetSubstructMatches(self.query, uniquify=False, maxMatches=10000):
            mapped = set(match)
            permitted = {match[i] for i in self.anchors.values()}
            if any(a.GetAtomicNum() == 7 and mol.GetAtomWithIdx(match[a.GetIdx()]).GetTotalNumHs() != 1
                   for a in self.core.GetAtoms()):
                continue
            boundaries = [(a.GetIdx(), n.GetIdx()) for a in mol.GetAtoms() if a.GetIdx() in mapped
                          for n in a.GetNeighbors() if n.GetIdx() not in mapped]
            if any(a not in permitted or mol.GetBondBetweenAtoms(a, b).GetBondType() != Chem.BondType.SINGLE
                   for a, b in boundaries):
                continue
            if len({a for a, _ in boundaries}) != len(boundaries):
                continue
            # Reject bridges between two sites (a core substructure match alone would accept them).
            exterior = set(range(mol.GetNumAtoms())) - mapped
            while exterior:
                component, pending = set(), [next(iter(exterior))]
                while pending:
                    i = pending.pop()
                    if i in component:
                        continue
                    component.add(i)
                    pending.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                                   if n.GetIdx() in exterior and n.GetIdx() not in component)
                exterior -= component
                if sum(b in component for _, b in boundaries) != 1:
                    break
            else:
                return True
        return False

    def classify(self, smiles):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or not mol.GetNumAtoms():
            return "invalid", None
        if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return "disconnected_or_dummy", mol
        if Chem.MolToSmiles(mol) == self.seed:
            return "seed_copy", mol
        if self.strict_match(mol):
            return "strict", mol
        if mol.HasSubstructMatch(self.relaxed):
            return "expanded_round2", mol
        return "outside_series", mol


def site_proposals(series, changes):
    sites = series.config["sites"]
    choices = list(itertools.product(*(s["fragments"] for s in sites)))
    random.Random(series.config["random_seed"]).shuffle(choices)
    for values in choices:
        fragments = {s["label"]: v for s, v in zip(sites, values)}
        changed = [s["name"] for s in sites if fragments[s["label"]] != s["original"]]
        if len(changed) != changes:
            continue
        base = {"changed_sites": changed, "fragments": fragments, "route_status": "graph_edit_only; synthesis_unvalidated"}
        try:
            yield dict(base, smiles=series.assemble(fragments))
        except (ValueError, RuntimeError) as exc:
            yield dict(base, smiles="", generation_error=str(exc))


def mol2mol_proposals(path, seed):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"SMILES", "Input_SMILES"}.issubset(reader.fieldnames or []):
            raise ValueError("Mol2Mol CSV requires SMILES and Input_SMILES for lineage verification")
        for index, row in enumerate(reader):
            if canonical_smiles(row["Input_SMILES"]) != seed:
                raise ValueError(f"Mol2Mol row {index}: parent differs from A-803467; refusing old demo inputs")
            yield {"smiles": row["SMILES"], "model_row": index, "model_metadata": row,
                   "route_status": "model_sample; synthesis_unvalidated"}


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["candidate_id", "smiles"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def run(config_path, output, mol2mol_csv=None):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    series = Series(config)
    budget = config["budget_per_method"]
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget_per_method must be positive")
    blocks = (config_path.parent / config["blocks"]).resolve()
    generators = {"site_single": lambda: site_proposals(series, 1), "site_pair": lambda: site_proposals(series, 2),
                  "amide_recombination": lambda: enumerate_library([series.seed], blocks, config["random_seed"])}
    if any(m not in generators for m in config["methods"]):
        raise ValueError("Unknown generation strategy")
    if mol2mol_csv:
        # Verify the entire file before creating any output or truncating its evaluation budget.
        imported = list(mol2mol_proposals(mol2mol_csv, series.seed))
        generators["mol2mol"] = lambda: iter(imported)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config_snapshot.json", config)
    save_json(output / "core_audit.json", {
        "strict_core_smarts": Chem.MolToSmarts(series.core),
        "attachment_core_atom_indices_zero_based": series.anchors,
        "secondary_amide_NH_required": True,
        "round2_core_smarts": config["round2_core_smarts"],
        "scope_basis": ["ROUND2.md: anilide-amide-furan core", "项目实现说明.md: aryl chlorine and methoxy local modifications"],
        "interpretation": "Full three-ring topology and named attachment sites are a conservative first-round design policy, not experimentally proven immutable pharmacophore.",
        "symmetric_sites": ["anilide_meta_a", "anilide_meta_b"],
        "site_labels_are_not_pdb_atom_numbers": True})
    (output / "seed.smi").write_text(series.seed + "\n")
    records, pool, summaries = [], {}, {}
    for method in [*config["methods"], *(["mol2mol"] if mol2mol_csv else [])]:
        proposals = list(itertools.islice(generators[method](), budget + 1))
        truncated = len(proposals) > budget
        rows = proposals[:budget]
        own_unique = set()
        counts = {}
        for idx, proposal in enumerate(rows):
            raw = proposal.get("smiles", "")
            status, mol = series.classify(raw)
            canonical = Chem.MolToSmiles(mol) if mol is not None else ""
            cid = candidate_id(canonical) if canonical else None
            counts[status] = counts.get(status, 0) + 1
            if cid:
                own_unique.add(cid)
            event = {"event_id": f"{config['series_id']}:{method}:{idx}", "series_id": config["series_id"],
                     "seed_id": config["seed_id"], "parent_smiles": series.seed, "method": method,
                     "raw_index": idx, **proposal, "raw_smiles": raw, "smiles": canonical,
                     "candidate_id": cid, "status": status}
            records.append(event)
            if status not in ("strict", "expanded_round2"):
                continue
            if cid not in pool:
                pool[cid] = {"candidate_id": cid, "smiles": canonical, "scope": status, "series_id": config["series_id"],
                             "methods": [], "event_ids": [], "mw": Descriptors.MolWt(mol),
                             "logp": Crippen.MolLogP(mol), "tpsa": rdMolDescriptors.CalcTPSA(mol),
                             "unassigned_stereocentres": sum(tag == "?" for _,tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
                             "geometry_status": "not_generated", "activity_status": "not_evaluated"}
            entry = pool[cid]
            entry["event_ids"].append(event["event_id"])
            if method not in entry["methods"]:
                entry["methods"].append(method)
        summaries[method] = {"processed_rows": len(rows), "unique_parseable": len(own_unique), "statuses": counts,
                             "budget": budget, "truncated": truncated,
                             "counting_scope": "emitted sanitized reaction products" if method == "amide_recombination" else "enumerated site assignments" if method.startswith("site_") else "imported raw sampling rows"}
    values = sorted(pool.values(), key=lambda r: r["candidate_id"])
    with (output / "lineage.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output / "all_proposals.csv", records)
    for scope, filename in [("strict", "candidates"), ("expanded_round2", "expanded_candidates")]:
        selected = [r for r in values if r["scope"] == scope]
        write_csv(output / f"{filename}.csv", selected)
        (output / f"{filename}.smi").write_text("".join(r["smiles"] + " " + r["candidate_id"] + "\n" for r in selected))
        with Chem.SDWriter(str(output / f"{filename}_2d.sdf")) as w:
            for row in selected:
                mol = Chem.MolFromSmiles(row["smiles"])
                AllChem.Compute2DCoords(mol)
                for k,v in row.items():
                    mol.SetProp(k, json.dumps(v) if isinstance(v,list) else str(v))
                w.write(mol)
    summary = {"series_id": config["series_id"], "methods": summaries, "strict_unique": sum(r["scope"] == "strict" for r in values),
               "expanded_unique": sum(r["scope"] != "strict" for r in values), "unique_with_multiple_methods": sum(len(r["methods"]) > 1 for r in values),
               "mol2mol_status": "imported_parent_verified" if mol2mol_csv else "awaiting_real_sampling",
               "rdkit_version": rdBase.rdkitVersion, "ranking": "none; IDs sorted for reproducible presentation",
               "blocks_sha256": hashlib.sha256(blocks.read_bytes()).hexdigest(), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
               "mol2mol_input_sha256": hashlib.sha256(Path(mol2mol_csv).read_bytes()).hexdigest() if mol2mol_csv else None,
               "property_filtering": "none; descriptors only", "sdf_coordinates": "2D depictions, not generated conformers or docking poses"}
    save_json(output / "summary.json", summary)
    # Representative small review sheet; complete library remains in SDF/CSV.
    shown = [series.seed] + [r["smiles"] for r in values if r["scope"] == "strict"][:11]
    legends = ["A-803467 reference"] + [r["candidate_id"] for r in values if r["scope"] == "strict"][:11]
    svg = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in shown], legends=legends, molsPerRow=3,
                              subImgSize=(420,250), useSVG=True)
    (output / "review.svg").write_text(svg, encoding="utf-8")
    depiction = Chem.MolFromSmiles(series.seed)
    match = depiction.GetSubstructMatch(series.query)
    labels = {match[i]: f"R{label}" for label,i in series.anchors.items()}
    for i,label in labels.items():
        depiction.GetAtomWithIdx(i).SetProp("atomNote", label)
    Draw.MolToFile(depiction, str(output / "seed_sites.png"), size=(1100,650), highlightAtoms=list(labels),
                   legend="A-803467: R1 distal para; R2/R3 anilide meta (symmetry-related)")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mol2mol-csv", type=Path)
    a = p.parse_args()
    run(a.config, a.output, a.mol2mol_csv)
