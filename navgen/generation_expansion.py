"""Expand one reviewed series and retain historical structures, without score ranking."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors

from .expansion_blocks import build_blocks, cross_proposals
from .generation import candidate_id
from .generation_audit import (read_batch, method_family, diagnose, describe_sites,
                               vocabulary_fragment, summarize, select_pilot, render_gallery)
from .series_generation import Series, site_proposals, mol2mol_proposals, save_json, write_csv


RETAIN = ("strict", "expanded_round2", "outside_series")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vocabulary_evidence(base_series, series, prior_rows):
    """Every added fragment must be witnessed at that same site in a reviewed strict structure."""
    observed = defaultdict(set)
    for row in prior_rows:
        if row["scope"] != "strict":
            continue
        observed[(1, row["R1"])].add(row["candidate_id"])
        for fragment in row["R2_R3_unordered"]:
            for label in (2, 3):
                observed[(label, fragment)].add(row["candidate_id"])
    original_vocab = {s["label"]: {vocabulary_fragment(f) for f in s["fragments"]} for s in base_series.config["sites"]}
    evidence = []
    for site in series.config["sites"]:
        seen = set()
        for fragment in site["fragments"]:
            attached = vocabulary_fragment(fragment)
            if attached in seen:
                raise ValueError("Duplicate canonical fragment in site vocabulary")
            seen.add(attached)
            old = attached in original_vocab[site["label"]]
            sources = sorted(observed[(site["label"], attached)])
            if not old and not sources:
                raise ValueError(f"New fragment lacks site-specific reviewed evidence: R{site['label']} {fragment}")
            evidence.append({"site": site["label"], "fragment": fragment or "H", "attached_fragment": attached,
                             "origin": "prior_vocabulary" if old else "reviewed_generated_structure",
                             "source_candidate_ids": sources, "activity_and_synthesis": "unvalidated"})
    return evidence


def make_pool(series, records, previous_ids):
    pool = {}
    for event in records:
        if event["status"] not in RETAIN:
            continue
        cid, smi = event["candidate_id"], event["smiles"]
        if cid not in pool:
            mol = Chem.MolFromSmiles(smi)
            info = diagnose(series, mol, event["status"])
            row = {"candidate_id": cid, "smiles": smi, "scope": event["status"], "series_id": series.config["series_id"],
                   "methods": [], "submethods": [], "event_ids": [], "sampling_batches": [], "is_new_vs_baseline": cid not in previous_ids,
                   "review_status": "new_unreviewed" if cid not in previous_ids else "retained_after_general_user_review",
                   "mw": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol), "tpsa": rdMolDescriptors.CalcTPSA(mol),
                   "unassigned_stereocentres": sum(tag == "?" for _, tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
                   "geometry_status": "not_generated_in_this_run", "activity_status": "not_evaluated", **info}
            if event["status"] == "strict":
                row.update(describe_sites(series, mol, info["core_match"]))
            pool[cid] = row
        row = pool[cid]
        if row["smiles"] != smi or row["scope"] != event["status"]:
            raise ValueError("Inconsistent candidate identity or scope")
        for key, val in (("methods", method_family(event["method"])), ("submethods", event["method"]), ("event_ids", event["event_id"])):
            if val not in row[key]:
                row[key].append(val)
        batch = event.get("model_metadata", {}).get("sampling_batch")
        if batch and batch not in row["sampling_batches"]:
            row["sampling_batches"].append(batch)
    return sorted(pool.values(), key=lambda r: r["candidate_id"])


def export_library(output, stem, rows):
    write_csv(output / f"{stem}.csv", rows)
    (output / f"{stem}.smi").write_text("".join(r["smiles"] + " " + r["candidate_id"] + "\n" for r in rows), encoding="utf-8")
    with Chem.SDWriter(str(output / f"{stem}_2d.sdf")) as writer:
        for row in rows:
            mol = Chem.MolFromSmiles(row["smiles"])
            AllChem.Compute2DCoords(mol)
            for key, value in row.items():
                mol.SetProp(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value))
            writer.write(mol)


def growth_metrics(records, previous_ids):
    local = {r["candidate_id"] for r in records if r.get("generation_phase") == "expansion"
             and r["method"] != "mol2mol" and r["status"] in RETAIN}
    seen, results = set(), []
    batches = defaultdict(list)
    for r in records:
        if r.get("generation_phase") == "expansion" and r["method"] == "mol2mol":
            batches[str(r["model_metadata"].get("sampling_batch", "1"))].append(r)
    for batch, entries in sorted(batches.items(), key=lambda item: int(item[0])):
        eligible = {r["candidate_id"] for r in entries if r["status"] in RETAIN}
        by_scope = {scope: len({r["candidate_id"] for r in entries if r["status"] == scope}) for scope in RETAIN}
        fragments = set()
        for r in entries:
            if r["status"] == "strict":
                fragments.update(r.get("site_fragment_tokens", []))
        results.append({"batch": batch, "raw_rows": len(entries), "unique_by_scope": by_scope,
                        "new_vs_previous_batches": len(eligible - seen),
                        "new_vs_previous_batches_and_baseline": len(eligible - seen - previous_ids),
                        "new_vs_previous_batches_baseline_and_local_expansion": len(eligible - seen - previous_ids - local),
                        "strict_site_fragment_types_in_batch": len(fragments),
                        "new_strict_site_fragment_types_vs_prior_batches": sorted(fragments - set().union(*(set(x.get("_fragments", [])) for x in results))),
                        "_fragments": sorted(fragments)})
        seen |= eligible
    for row in results:
        row.pop("_fragments")
    return results


def run(config_path, baseline, output, mol2mol_csv=None, images=True):
    config_path, baseline, output = map(lambda p: Path(p).resolve(), (config_path, baseline, output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    series = Series(config)
    base_series, old_events, prior_rows = read_batch(baseline)
    if (series.seed != base_series.seed or config["attachment_scaffold"] != base_series.config["attachment_scaffold"]
            or config["round2_core_smarts"] != base_series.config["round2_core_smarts"]):
        raise ValueError("Expansion must preserve the audited seed/core/site boundaries")
    if config["retain_scopes"] != list(RETAIN):
        raise ValueError("All previously retained structural scopes must remain available")
    evidence = vocabulary_evidence(base_series, series, prior_rows)
    block_path = (config_path.parent / config["blocks"]).resolve()
    blocks, diagnostics = build_blocks(block_path, prior_rows, config["derive_blocks_from_scopes"])
    source_hash = sha(baseline / "lineage.jsonl")
    config_hash = sha(config_path)
    previous_ids = {r["candidate_id"] for r in prior_rows if r["scope"] in RETAIN}
    records = [{**r, "source_event_id": r["event_id"], "event_id": "history:" + source_hash[:12] + ":" + r["event_id"],
                "generation_phase": "historical", "source_lineage_sha256": source_hash} for r in old_events]
    methods = {"site_single": lambda: site_proposals(series, 1), "site_pair": lambda: site_proposals(series, 2),
               "site_triple": lambda: site_proposals(series, 3), "amide_cross_product": lambda: cross_proposals(blocks, series.seed)}
    sampling_provenance = {}
    if mol2mol_csv:
        mol2mol_csv = Path(mol2mol_csv).resolve()
        proposals = list(mol2mol_proposals(mol2mol_csv, series.seed))
        sampling_provenance = {"file": str(mol2mol_csv), "sha256": sha(mol2mol_csv)}
        manifest = mol2mol_csv.with_name("manifest.json")
        if manifest.exists():
            data = json.loads(manifest.read_text())
            if data["status"] != "completed" or data.get("combined_sha256", data.get("output_sha256")) != sha(mol2mol_csv):
                raise ValueError("Sampling manifest is incomplete or CSV hash differs")
            sampling_provenance["manifest"] = data
        methods["mol2mol"] = lambda: iter(proposals)
    counters, classification_cache = {}, {}
    for method, generator in methods.items():
        budget = config["budgets"][method]
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Every method requires a positive integer budget")
        print(f"Generating {method}, record budget {budget}", flush=True)
        batch = list(itertools.islice(generator(), budget + 1))
        if len(batch) > budget:
            raise ValueError(f"{method} exceeds explicit budget; adjust budget or sampling before running (no silent truncation)")
        unique, statuses = set(), Counter()
        for index, proposal in enumerate(batch):
            raw = proposal.get("smiles", "")
            if raw not in classification_cache:
                classification_cache[raw] = series.classify(raw)
            status, mol = classification_cache[raw]
            smi = Chem.MolToSmiles(mol) if mol is not None else ""
            cid = candidate_id(smi) if smi else None
            if cid:
                unique.add(cid)
            statuses[status] += 1
            record = {**proposal, "event_id": f"expansion:{config_hash[:12]}:{method}:{index}", "series_id": config["series_id"],
                      "seed_id": config["seed_id"], "parent_smiles": series.seed, "method": method, "raw_index": index,
                      "raw_smiles": raw, "smiles": smi, "candidate_id": cid, "status": status, "generation_phase": "expansion"}
            if method == "mol2mol":
                record["sampling_input_sha256"] = sampling_provenance["sha256"]
                if status == "strict":
                    info = diagnose(series, mol, status)
                    sites = describe_sites(series, mol, info["core_match"])
                    record["site_fragment_tokens"] = ["R1:" + sites["R1"]] + ["R2/R3:" + f for f in sites["R2_R3_unordered"]]
            records.append(record)
        counters[method] = {"raw_rows": len(batch), "unique_parseable_including_seed": len(unique), "statuses": dict(statuses),
                            "budget": budget, "truncated": False,
                            "counting_scope": "reaction products including failures" if method == "amide_cross_product" else "raw sampling rows" if method == "mol2mol" else "site assignments including symmetry duplicates"}
    if len({r["event_id"] for r in records}) != len(records):
        raise ValueError("Duplicate qualified event IDs")
    pool = make_pool(series, records, previous_ids)
    if not previous_ids.issubset({r["candidate_id"] for r in pool}):
        raise ValueError("Historical candidate lost during expansion")
    growth = growth_metrics(records, previous_ids)
    summary = {"series_id": config["series_id"], "rdkit_version": rdBase.rdkitVersion,
               "historical_raw_rows": len(old_events), "new_raw_rows": len(records) - len(old_events), "total_raw_rows": len(records),
               "previous_unique_retained": len(previous_ids), "total_unique_retained": len(pool),
               "new_unique_vs_baseline": sum(r["is_new_vs_baseline"] for r in pool),
               "unique_by_scope": dict(Counter(r["scope"] for r in pool)),
               "new_unique_by_scope": dict(Counter(r["scope"] for r in pool if r["is_new_vs_baseline"])),
               "methods_new_run": counters, "virtual_blocks": dict(Counter(r["role"] for r in blocks)),
               "reaction_pairs": len([r for r in blocks if r["role"] == "acid"]) * len([r for r in blocks if r["role"] == "amine"]),
               "strict_changed_site_counts": dict(Counter(str(r["changed_site_count"]) for r in pool if r["scope"] == "strict")),
               "baseline_lineage_sha256": source_hash, "config_sha256": config_hash, "base_blocks_sha256": sha(block_path),
               "sampling": sampling_provenance or {"status": "not_imported"}, "scoring": "none",
               "code_sha256": {name: sha(Path(__file__).with_name(name)) for name in ("generation_expansion.py", "expansion_blocks.py", "generation_audit.py", "series_generation.py")},
               "ranking": "none; IDs only", "sdf_coordinates": "2D depiction only", "review_context": config["review_context"]}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config_snapshot.json", config)
    save_json(output / "baseline_config_snapshot.json", base_series.config)
    save_json(output / "summary.json", summary)
    save_json(output / "virtual_blocks.json", blocks)
    write_csv(output / "virtual_blocks.csv", blocks)
    save_json(output / "block_diagnostics.json", diagnostics)
    write_csv(output / "vocabulary_evidence.csv", evidence)
    with (output / "lineage.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(output / "all_proposals.csv", records)
    write_csv(output / "all_retained.csv", pool)
    for scope, stem in (("strict", "candidates"), ("expanded_round2", "expanded_candidates"), ("outside_series", "outside_candidates")):
        print(f"Exporting {scope}", flush=True)
        export_library(output, stem, [r for r in pool if r["scope"] == scope])
    # Supply the same complete-record audit schema as the smaller batch.
    for row in pool:
        row["methods"].sort()
        row["contribution"] = "exclusive:" + row["methods"][0] if len(row["methods"]) == 1 else "shared:" + "+".join(row["methods"])
        if "change_description" not in row:
            from .generation_audit import REASONS
            row["change_description"] = "；".join(REASONS[r] for r in row["constraint_reasons"])
    save_json(output / "audit_records.json", pool)
    seed_methods = sorted({method_family(r["method"]) for r in records if r["status"] == "seed_copy"})
    seed_summary_row = [{"scope": "seed_copy", "methods": seed_methods, "constraint_reasons": []}] if seed_methods else []
    contribution_summary = summarize(records, pool + seed_summary_row)
    write_csv(output / "method_contributions.csv", contribution_summary["contributions"])
    save_json(output / "contributions.json", contribution_summary)
    save_json(output / "mol2mol_batch_growth.json", growth)
    write_csv(output / "mol2mol_batch_growth.csv", growth)
    new_pool = [r for r in pool if r["is_new_vs_baseline"]]
    pilot = select_pilot(series, new_pool, strict_count=12, expanded_count=6)
    save_json(output / "pilot_selection.json", {"source": str(output), "policy": "new structures; coverage then diversity; no activity ranking", "candidates": pilot})
    if images:
        chosen = {r["candidate_id"] for r in pilot if r["scope"] != "reference"}
        selected = [r for r in pool if r["candidate_id"] in chosen]
        # Include up to three newly generated series-outside boundaries in the visual review only.
        selected += [r for r in new_pool if r["scope"] == "outside_series"][:3]
        index = render_gallery(series, selected, output / "review_first", "new_representatives")
        write_csv(output / "review_first_index.csv", index)
    write_report(output, summary, growth)
    return summary


def write_report(output, summary, growth):
    lines = ["# A-803467 生成扩容结果", "", "本轮不做活性/docking 排名，不训练模型；保留旧结构及完整来源。", "",
             f"原有 {summary['previous_unique_retained']} 个非母体结构全部保留。合并后 {summary['total_unique_retained']} 个，净新增 {summary['new_unique_vs_baseline']} 个。", "",
             "| 分类 | 合并唯一结构 | 本轮净新增 |", "|---|---:|---:|"]
    for scope in RETAIN:
        lines.append(f"| {scope} | {summary['unique_by_scope'].get(scope,0)} | {summary['new_unique_by_scope'].get(scope,0)} |")
    lines += ["", f"虚拟砌块：{summary['virtual_blocks']}；酸×胺共 {summary['reaction_pairs']} 对。每个砌块记录原始砌块 ID 或拆分来源候选 ID，尚未核实采购、保护基、反应条件及合成选择性。",
              "", f"主库改变位点数分布：{summary['strict_changed_site_counts']}。单点、双点、三点分组保留；R2/R3 对称去重。",
              "", "## Mol2Mol 分批收益", "", "| 批次 | 原始条数 | 相对旧库及前批新增 | 再扣除本轮本地扩容后新增 |", "|---|---:|---:|---:|"]
    for row in growth:
        lines.append(f"| {row['batch']} | {row['raw_rows']} | {row['new_vs_previous_batches_and_baseline']} | {row['new_vs_previous_batches_baseline_and_local_expansion']} |")
    lines += ["", "这是原始采样的结构增量统计，不是活性或模型优劣比较；完整主库/扩展/系列外统计见 mol2mol_batch_growth.json。", "",
              "## 查看文件", "", "- review_first/：新增结构代表与 A-803467 对照，Windows 照片可直接打开。不是活性优选。",
              "- all_retained.csv：所有唯一候选，标明是否本轮新增。",
              "- candidates / expanded_candidates / outside_candidates 的 SMI、CSV、2D SDF：分别为主库、扩展库、系列外保留集。",
              "- lineage.jsonl：含旧记录、本轮所有尝试与模型重复输出；虚拟反应产物可经两个 block ID 追溯来源。",
              "- vocabulary_evidence.csv：新增位点片段的既有候选来源，按连接位置核验。",
              "- pilot_selection.json：供后续三维准备的新增代表集；本轮尚未准备全库 3D。", "",
              "现有候选被用户概览后保留，不等于逐个通过活性/毒性/合成审查；本轮新结构仍标为未审阅。核心边界保持原定义，系列外结构单列。", ""]
    (output / "扩容结果.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mol2mol-csv", type=Path)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.baseline, args.output, args.mol2mol_csv, not args.no_images), ensure_ascii=False, indent=2))
