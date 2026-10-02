"""Audit an immutable generation batch; compare structures without activity ranking."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Draw, rdDepictor, rdFMCS, rdFingerprintGenerator

from .generation import candidate_id
from .series_generation import Series, save_json, write_csv


METHODS = ("site_enumeration", "amide_recombination", "mol2mol")
METHOD_NAMES = dict(zip(METHODS, ("位点枚举", "酰胺重组", "Mol2Mol")))
SCOPES = {"strict": "主库", "expanded_round2": "扩展库", "outside_series": "系列外", "seed_copy": "母体重复"}
REASONS = {
    "full_core_missing": "完整三环核心不匹配",
    "round2_core_missing": "旧 ROUND2 核心也不匹配",
    "amide_NH_changed": "二级酰胺 NH 改变",
    "offsite_substitution": "声明位点以外出现取代",
    "non_single_attachment": "取代基连接键不是单键",
    "multiple_attachments": "同一位点出现多条外接键",
    "bridge_or_fusion": "外部片段跨位点连接或形成稠环",
    "disconnected_component": "存在独立组分",
    "disconnected_or_dummy": "多组分或虚原子",
    "invalid": "结构解析失败",
}


def method_family(method):
    if method in ("site_single", "site_pair", "site_triple"):
        return "site_enumeration"
    if method == "amide_cross_product":
        return "amide_recombination"
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return method


def outside_components(mol, match):
    mapped = set(match)
    remaining = set(range(mol.GetNumAtoms())) - mapped
    components = []
    while remaining:
        todo, component = [min(remaining)], set()
        while todo:
            idx = todo.pop()
            if idx in component:
                continue
            component.add(idx)
            todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                        if n.GetIdx() in remaining and n.GetIdx() not in component)
        remaining -= component
        edges = sorted((n.GetIdx(), idx) for idx in component
                       for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in mapped)
        components.append((component, edges))
    return components


def match_reasons(series, mol, match):
    """Explain the existing classifier, considering all symmetry-related mappings."""
    permitted = {match[i] for i in series.anchors.values()}
    reasons, offsites = set(), []
    for atom in series.core.GetAtoms():
        if atom.GetAtomicNum() == 7 and mol.GetAtomWithIdx(match[atom.GetIdx()]).GetTotalNumHs() != 1:
            reasons.add("amide_NH_changed")
    edges = []
    for _, boundary in outside_components(mol, match):
        edges.extend(boundary)
        if len(boundary) > 1:
            reasons.add("bridge_or_fusion")
        if not boundary:
            reasons.add("disconnected_component")
    inverse = {idx: core_idx for core_idx, idx in enumerate(match)}
    for a, b in edges:
        if a not in permitted:
            # N substitution is already reported precisely as NH change.
            if series.core.GetAtomWithIdx(inverse[a]).GetAtomicNum() != 7:
                reasons.add("offsite_substitution")
                offsites.append(inverse[a])
        if mol.GetBondBetweenAtoms(a, b).GetBondType() != Chem.BondType.SINGLE:
            reasons.add("non_single_attachment")
    if len({a for a, _ in edges}) != len(edges):
        reasons.add("multiple_attachments")
    return sorted(reasons), sorted(set(offsites))


def diagnose(series, mol, status):
    if status in ("invalid", "disconnected_or_dummy"):
        return {"constraint_reasons": [status], "core_match": [], "offsite_core_atoms": []}
    matches = mol.GetSubstructMatches(series.query, uniquify=False, maxMatches=10000)
    if not matches:
        reasons = ["full_core_missing"]
        if not mol.HasSubstructMatch(series.relaxed):
            reasons.append("round2_core_missing")
        return {"constraint_reasons": reasons, "core_match": [], "offsite_core_atoms": []}
    attempts = [(match_reasons(series, mol, m), m) for m in matches]
    # Do not combine violations from alternative mappings: report a least-violating mapping.
    (reasons, offsites), match = min(attempts, key=lambda item: (len(item[0][0]), item[0], item[1]))
    if (not reasons) != (status in ("strict", "seed_copy")):
        raise ValueError("Constraint explanation disagrees with the source classifier")
    return {"constraint_reasons": reasons, "core_match": list(match), "offsite_core_atoms": offsites}


def attached_fragment(mol, edge):
    a, b = edge
    cut = Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(a, b).GetIdx()], dummyLabels=[(0, 0)])
    mappings = []
    fragments = Chem.GetMolFrags(cut, asMols=True, fragsMolAtomMapping=mappings)
    for fragment, mapping in zip(fragments, mappings):
        if b in mapping:
            for atom in fragment.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomMapNum(1)
            return Chem.MolToSmiles(fragment)
    raise ValueError("Missing substituent after bond cut")


def vocabulary_fragment(smiles):
    if not smiles:
        return "H"
    rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(1)
    idx = rw.AddAtom(dummy)
    rw.AddBond(0, idx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def site_fragments(series, mol, match):
    groups = {label: "H" for label in series.anchors}
    labels = {match[i]: label for label, i in series.anchors.items()}
    for _, edges in outside_components(mol, match):
        if len(edges) == 1 and edges[0][0] in labels:
            groups[labels[edges[0][0]]] = attached_fragment(mol, edges[0])
    return groups


def describe_sites(series, mol, match):
    groups = site_fragments(series, mol, match)
    # R2/R3 are a symmetry-equivalent unordered pair, never two distinct SAR positions.
    pair = sorted((groups[2], groups[3]))
    original = {k: vocabulary_fragment(v) for k, v in series.original.items()}
    vocab = {s["label"]: {vocabulary_fragment(f) for f in s["fragments"]} for s in series.config["sites"]}
    novel = sorted({f"R1={groups[1]}"} - {f"R1={f}" for f in vocab[1]})
    novel += sorted({f"R2/R3={f}" for f in pair if f not in vocab[2] | vocab[3]})
    changed = sum(groups[k] != original[k] for k in groups)
    return {"R1": groups[1], "R2_R3_unordered": pair, "changed_site_count": changed,
            "outside_enumerator_vocabulary": novel,
            "change_description": f"R1={groups[1]}; R2/R3={','.join(pair)}; 改变 {changed} 个位点"}


def read_batch(source):
    config_bytes = (source / "config_snapshot.json").read_bytes()
    config = json.loads(config_bytes)
    series = Series(config)
    events = [json.loads(line) for line in (source / "lineage.jsonl").read_text(encoding="utf-8").splitlines() if line]
    unique, event_ids, status_cache = {}, set(), {}
    for event in events:
        if event["event_id"] in event_ids:
            raise ValueError("Duplicate event ID in source")
        event_ids.add(event["event_id"])
        if event["parent_smiles"] != series.seed:
            raise ValueError("Source lineage parent differs from config")
        smi = event["smiles"]
        if smi not in status_cache:
            status_cache[smi] = series.classify(smi)
        status, mol = status_cache[smi]
        if status != event["status"]:
            raise ValueError(f"Source classification drift: {event['event_id']}")
        if mol is not None and event["candidate_id"] != candidate_id(Chem.MolToSmiles(mol)):
            raise ValueError("Source candidate ID does not match structure")
        key = event["candidate_id"] or event["event_id"]
        row = unique.setdefault(key, {"candidate_id": key, "smiles": smi, "scope": status,
                                      "methods": [], "submethods": [], "event_ids": []})
        if row["smiles"] != smi:
            raise ValueError("Candidate ID collision")
        for field, val in (("methods", method_family(event["method"])), ("submethods", event["method"]), ("event_ids", event["event_id"])):
            if val not in row[field]:
                row[field].append(val)
    rows = sorted(unique.values(), key=lambda r: r["candidate_id"])
    for row in rows:
        row["methods"].sort()
        row["contribution"] = "exclusive:" + row["methods"][0] if len(row["methods"]) == 1 else "shared:" + "+".join(row["methods"])
        mol = Chem.MolFromSmiles(row["smiles"]) if row["smiles"] else None
        row.update(diagnose(series, mol, row["scope"]))
        if row["scope"] in ("strict", "seed_copy"):
            row.update(describe_sites(series, mol, row["core_match"]))
        else:
            row["change_description"] = "；".join(REASONS[r] for r in row["constraint_reasons"])
        row["raw_occurrences"] = len(row["event_ids"])
    return series, events, rows


def summarize(events, rows):
    contributions = []
    for method in METHODS:
        ev = [e for e in events if method_family(e["method"]) == method]
        own = [r for r in rows if method in r["methods"]]
        for scope in ("strict", "expanded_round2", "outside_series", "seed_copy"):
            chosen = [r for r in own if r["scope"] == scope]
            contributions.append({"method": method, "scope": scope,
                                  "raw_rows": sum(e["status"] == scope for e in ev),
                                  "unique_structures": len(chosen),
                                  "exclusive_structures": sum(len(r["methods"]) == 1 for r in chosen),
                                  "shared_structures": sum(len(r["methods"]) > 1 for r in chosen)})
    reasons = []
    for scope in ("expanded_round2", "outside_series"):
        counts = Counter(reason for r in rows if r["scope"] == scope for reason in r["constraint_reasons"])
        reasons.extend({"scope": scope, "reason": k, "description": REASONS[k], "unique_structures": v} for k, v in sorted(counts.items()))
    return {"raw_records": len(events), "unique_by_scope": dict(Counter(r["scope"] for r in rows)),
            "contributions": contributions, "constraint_counts_multilabel": reasons,
            "overlap_patterns": {scope: dict(Counter("+".join(r["methods"]) for r in rows if r["scope"] == scope))
                                 for scope in ("strict", "expanded_round2", "outside_series")},
            "strict_changed_site_counts": dict(Counter(str(r["changed_site_count"]) for r in rows if r["scope"] == "strict")),
            "strict_outside_vocabulary": sum(bool(r.get("outside_enumerator_vocabulary")) for r in rows if r["scope"] == "strict"),
            "invalid_rows": sum(e["status"] == "invalid" for e in events)}


def select_pilot(series, rows, strict_count=10, expanded_count=5):
    """Coverage first, fingerprint diversity as a tie breaker; no efficacy ranking."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    selected = [{"candidate_id": candidate_id(series.seed), "smiles": series.seed, "scope": "reference",
                 "methods": [], "event_ids": [], "selection_reason": "A-803467 reference"}]
    selected_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(series.seed))]
    for scope, count in (("strict", strict_count), ("expanded_round2", expanded_count)):
        pool = [r for r in rows if r["scope"] == scope]
        covered = set()
        def tokens(row):
            return ({"origin:" + row["contribution"]} | {"method:" + m for m in row["methods"]}
                    | {"reason:" + r for r in row["constraint_reasons"]}
                    | ({"edits:" + str(row["changed_site_count"])} if "changed_site_count" in row else set())
                    | {"fragment:" + f for f in row.get("outside_enumerator_vocabulary", [])})
        fps = {r["candidate_id"]: fpgen.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in pool}
        for _ in range(min(count, len(pool))):
            scores = [(len(tokens(r) - covered), 1 - max(DataStructs.BulkTanimotoSimilarity(fps[r["candidate_id"]], selected_fps)), r["candidate_id"], r) for r in pool]
            _, distance, _, row = max(scores, key=lambda x: (x[0], x[1], x[2]))
            gained = sorted(tokens(row) - covered)
            selected.append({k: row[k] for k in ("candidate_id", "smiles", "scope", "methods", "event_ids")}
                            | {"selection_reason": gained or ["fingerprint diversity tie breaker"], "selection_min_distance": distance})
            covered |= tokens(row)
            selected_fps.append(fps[row["candidate_id"]])
            pool.remove(row)
    return selected


def comparison_molecules(series, row):
    ref, mol = Chem.MolFromSmiles(series.seed), Chem.MolFromSmiles(row["smiles"])
    AllChem.Compute2DCoords(ref)
    ref_match = ref.GetSubstructMatch(series.query)
    match = row["core_match"]
    if match:
        mapping = list(zip(ref_match, match))
        rdDepictor.GenerateDepictionMatching2DStructure(mol, ref, mapping)
        highlights, ref_highlights = set(), set()
        original = {k: vocabulary_fragment(v) for k, v in series.original.items()}
        groups = site_fragments(series, mol, match)
        for label, core_idx in series.anchors.items():
            if groups[label] != original[label]:
                highlights.add(match[core_idx])
                ref_highlights.add(ref_match[core_idx])
        permitted = {match[i]: k for k, i in series.anchors.items()}
        for component, edges in outside_components(mol, match):
            if (len(edges) != 1 or edges[0][0] not in permitted
                    or groups[permitted[edges[0][0]]] != original[permitted[edges[0][0]]]):
                highlights |= component | {a for a, _ in edges}
        for component, edges in outside_components(ref, ref_match):
            if any(a in ref_highlights for a, _ in edges):
                ref_highlights |= component
        alignment = "完整核心对齐；橙色为变化片段及连接位置"
    else:
        mcs = rdFMCS.FindMCS([ref, mol], timeout=3, ringMatchesRingOnly=True, completeRingsOnly=True,
                            bondCompare=rdFMCS.BondCompare.CompareOrderExact)
        query = Chem.MolFromSmarts(mcs.smartsString)
        a, b = ref.GetSubstructMatch(query), mol.GetSubstructMatch(query)
        if not a or mcs.canceled:
            AllChem.Compute2DCoords(mol)
            return ref, mol, [], [], "核心已变；独立绘图，未建立可靠原子对应"
        rdDepictor.GenerateDepictionMatching2DStructure(mol, ref, list(zip(a, b)))
        ref_highlights, highlights = set(range(ref.GetNumAtoms())) - set(a), set(range(mol.GetNumAtoms())) - set(b)
        for obj, marked in ((ref, ref_highlights), (mol, highlights)):
            marked.update(n.GetIdx() for i in list(marked) for n in obj.GetAtomWithIdx(i).GetNeighbors())
        alignment = "最大公共子结构近似对齐；橙色为差异区域（对应可能不唯一）"
    return ref, mol, sorted(ref_highlights), sorted(highlights), alignment


def font(size):
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def wrapped_text(draw, xy, text, size, max_width, fill="#222222"):
    f, x, y, line = font(size), xy[0], xy[1], ""
    for char in text:
        if draw.textlength(line + char, font=f) > max_width:
            draw.text((x, y), line, font=f, fill=fill)
            y += size + 6
            line = ""
        line += char
    draw.text((x, y), line, font=f, fill=fill)
    return y + size + 6


def render_mol(mol, highlights):
    drawer = Draw.MolDraw2DCairo(770, 285)
    opt = drawer.drawOptions()
    opt.fixedBondLength = 23
    opt.bondLineWidth = 2
    colors = {idx: (1.0, 0.72, 0.35) for idx in highlights}
    bonds = [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in colors and b.GetEndAtomIdx() in colors]
    drawer.DrawMolecule(mol, highlightAtoms=highlights, highlightAtomColors=colors,
                        highlightBonds=bonds, highlightBondColors={b: (1.0, 0.72, 0.35) for b in bonds})
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def render_gallery(series, rows, output, group_override=None):
    output.mkdir()
    groups = defaultdict(list)
    for row in rows:
        if row["scope"] in ("invalid", "seed_copy", "disconnected_or_dummy"):
            continue
        if row["scope"] == "strict":
            group = "strict_shared" if len(row["methods"]) > 1 else "strict_" + row["methods"][0]
        else:
            # A molecule appears once; all other violations remain listed on its card and in CSV.
            priority = ("round2_core_missing", "full_core_missing", "amide_NH_changed", "bridge_or_fusion", "offsite_substitution")
            reason = next((r for r in priority if r in row["constraint_reasons"]), row["constraint_reasons"][0])
            group = row["scope"] + "_" + reason
        groups[group_override or group].append(row)
    index = []
    for group, members in sorted(groups.items()):
        pages = (len(members) + 2) // 3
        print(f"Rendering {group}: {len(members)} structures", flush=True)
        for page in range(pages):
            batch = members[page * 3:page * 3 + 3]
            canvas = Image.new("RGB", (1600, 125 + len(batch) * 475), "white")
            draw = ImageDraw.Draw(canvas)
            wrapped_text(draw, (22, 15), f"A-803467 结构对照 | {group} | {page+1}/{pages}", 26, 1550)
            draw.text((24, 63), "左：母体   右：候选   |   按结构来源/约束分组；编号不是活性排名", font=font(23), fill="#555555")
            filename = f"{group}_{page+1:02d}.png"
            for i, row in enumerate(batch):
                y = 115 + 475 * i
                ref, mol, rh, mh, alignment = comparison_molecules(series, row)
                label = " + ".join(METHOD_NAMES[m] for m in row["methods"])
                draw.text((22, y), f"{row['candidate_id']}  |  {SCOPES[row['scope']]}  |  {label}", font=font(25), fill="#142940")
                canvas.paste(render_mol(ref, rh), (15, y + 37))
                canvas.paste(render_mol(mol, mh), (810, y + 37))
                end = wrapped_text(draw, (24, y + 325), row["change_description"], 22, 1540)
                wrapped_text(draw, (24, end), alignment, 20, 1540, "#555555")
                draw.line((20, y + 459, 1580, y + 459), fill="#dddddd", width=2)
                index.append({"candidate_id": row["candidate_id"], "scope": row["scope"], "methods": row["methods"],
                              "image": output.name + "/" + filename, "row_on_page": i + 1,
                              "constraint_reasons": row["constraint_reasons"], "alignment": alignment})
            canvas.save(output / filename)
    return index


def write_report(output, summary, rows, pilot):
    lines = ["# A-803467 现有生成空间审计", "", "基于已有批次，不重新生成、不做活性或 docking 排名。独有只指本批次三种方法之间的独有，不是文献/专利新颖性。", "",
             "| 方法 | 主库唯一 | 主库独有 | 主库共有 | 扩展库唯一 | 扩展库独有 | 系列外唯一 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        by_scope = {r["scope"]: r for r in summary["contributions"] if r["method"] == method}
        a, b, c = (by_scope[s] for s in ("strict", "expanded_round2", "outside_series"))
        lines.append(f"| {METHOD_NAMES[method]} | {a['unique_structures']} | {a['exclusive_structures']} | {a['shared_structures']} | {b['unique_structures']} | {b['exclusive_structures']} | {c['unique_structures']} |")
    lines += ["", "共有数按方法分别计数，不能相加当总数；精确交集见 summary.json。单点/双点合为位点枚举，操作记录仍完整保留。反应分支的输入已经 sanitize；三种方法预算和输入记录口径不同，不据此比较模型优劣。", "", "## 核心约束排除了什么", "",
              "| 去向 | 原因 | 唯一结构数 |", "|---|---|---:|"]
    lines.extend(f"| {SCOPES[r['scope']]} | {r['description']} | {r['unique_structures']} |" for r in summary["constraint_counts_multilabel"])
    lines += ["", "原因可多选，不能相加当总数。对称映射采用违反约束最少的一种，不合并不同映射的理由。主库规则是本轮设计边界，不是实验证明的不可变药效团；系列外结构仍保留，未混入主库。", "",
              "## 主库的实际化学变化", "",
              f"- 改变位点数分布：{summary['strict_changed_site_counts']}。R2/R3 按无序对处理。",
              f"- 有 {summary['strict_outside_vocabulary']} 个主库结构含本地枚举词表以外的取代基。它们仍可能满足核心/位点约束；词表不等于全局白名单。",
              "- 枚举分支只生成单点/双点操作；主库判定只约束可变位置，不限制同时变化的位点数。模型生成的三点变化单独标注，不应称为单点优化。",
              "- 位点详情及新片段见 structure_audit.csv；星号表示与核心连接的位置。", "",
              "## 先判断哪些变化值得保留", "",
              "1. 在 review_decisions.csv 对代表和边界结构填写：保留作假设 / 暂缓 / 建议放宽哪个位点或核心；补充文献、SAR 或可合成性依据。无需按图册编号选优。",
              "2. 先检查 Mol2Mol/反应独有的新片段是否提供有意义的极性、体积或连接方式变化；若有明确目标，再将选中的片段补进词表或寻找相应真实砌块。",
              "3. 对位点迁移、N-取代、稠环和换核心分开决定；放宽时另建探索分支，保留现有主库作为对照。不要一次解除所有约束。",
              "4. 只有当目标确实需要三维形状/药效团或骨架探索，并有合适参考结构时，再启动 ShEPhERD。", "",
              "## 图册与三维准备试跑", "",
              "- structure_comparison/：所有主库、扩展库及系列外唯一结构与母体的完整 PNG 对照；Windows 照片可直接打开。",
              "- structure_index.csv：按候选 ID 查图片和页内行号；一结构一处展示，多条排除原因同时标出。",
              "- review_first/：优先查看的 15 个代表对照图（5 页），不是活性优选。其余结构仍完整保留在全图册。",
              f"- pilot_selection.json：母体 + {len(pilot)-1} 个代表。按来源/变化覆盖选取，指纹多样性仅作同等覆盖时的决胜条件，不是活性优选。",
              "- 三维试跑保留输入化学状态；未做 pH 质子化预测或全面互变异构体枚举。通过几何检查不能证明结合、活性或合成可行性。", ""]
    (output / "审计结论.md").write_text("\n".join(lines), encoding="utf-8")


def run(source, output, images=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    series, events, rows = read_batch(source)
    summary = summarize(events, rows)
    pilot = select_pilot(series, rows)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("config_snapshot.json", "lineage.jsonl", "summary.json")}
    summary.update({"source": str(source), "input_sha256": inputs, "rdkit_version": rdBase.rdkitVersion,
                    "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "scoring": "none", "novelty_scope": "this batch only", "pilot_count": len(pilot)})
    save_json(output / "summary.json", summary)
    save_json(output / "audit_records.json", rows)
    save_json(output / "pilot_selection.json", {"source": str(source), "input_sha256": inputs,
                                               "policy": "coverage then fingerprint diversity; not activity selection", "candidates": pilot})
    write_csv(output / "structure_audit.csv", rows)
    write_csv(output / "method_contributions.csv", summary["contributions"])
    write_csv(output / "constraint_reasons.csv", summary["constraint_counts_multilabel"])
    write_csv(output / "review_decisions.csv", [{"candidate_id": r["candidate_id"], "scope": r["scope"],
                "methods": r["methods"], "change_description": r["change_description"],
                "selected_for_geometry_pilot": r["candidate_id"] in {p["candidate_id"] for p in pilot},
                "review_decision": "pending", "rationale_or_evidence": "", "proposed_scope_change": ""}
               for r in rows if r["scope"] not in ("seed_copy", "invalid")])
    if images:
        index = render_gallery(series, rows, output / "structure_comparison")
        write_csv(output / "structure_index.csv", index)
        summary["depicted_structures"] = len(index)
        summary["png_pages"] = len({r["image"] for r in index})
        selected_ids = {p["candidate_id"] for p in pilot if p["scope"] != "reference"}
        preview = render_gallery(series, [r for r in rows if r["candidate_id"] in selected_ids], output / "review_first", "pilot_representatives")
        write_csv(output / "review_first_index.csv", preview)
        save_json(output / "summary.json", summary)
    write_report(output, summary, rows, pilot)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, not args.no_images), ensure_ascii=False, indent=2))
