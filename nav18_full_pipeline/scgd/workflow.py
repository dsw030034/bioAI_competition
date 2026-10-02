"""Input lineage, preparation audit and resumable batches around the original score."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import io
import os
from pathlib import Path
import sys

from .ligand_quality import state_record
from .runtime import (BASE, code_hashes, read_json, replace_file, resolve_vina, sha256,
                      versions, work_manifest, write_json)


def read_entries(path, smiles_column='smiles', label_column=None):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in ('.smi', '.txt'):
        rows = []
        for i, line in enumerate(text.splitlines()):
            if line.strip() and not line.lstrip().startswith('#'):
                parts = line.split()
                rows.append({'candidate_id': parts[1] if len(parts) > 1 else f'mol_{i}',
                             'smiles': parts[0]})
    else:
        reader = csv.DictReader(io.StringIO(text, newline=''))
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate CSV column names')
        if smiles_column not in headers or (label_column and label_column not in headers):
            raise ValueError('Requested SMILES/label column is absent')
        rows = []
        for i, row in enumerate(reader):
            if None in row:
                raise ValueError(f'Malformed CSV row {i + 2}')
            label_key = label_column or ('candidate_id' if 'candidate_id' in headers else None)
            label = (row.get(label_key) or '').strip() if label_key else f'mol_{i}'
            # Empty structures remain explicit failure records instead of silently disappearing.
            rows.append({**row, 'candidate_id': label, 'smiles': (row.get(smiles_column) or '').strip()})
    ids = [r['candidate_id'] for r in rows]
    if not rows or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Input is empty or has empty/duplicate candidate IDs')
    return rows


def protocol(score, receptors, args):
    return {'receptors': {name: {'sha256': sha256(path), 'center': list(score.RECEPTORS[name]['center']),
                                  'size': score.RECEPTORS[name]['size']}
                          for name, path in receptors.items()},
            'vina_sha256': sha256(resolve_vina(args.vina)), 'exhaustiveness': args.exhaustiveness,
            'num_modes': args.num_modes, 'cpu': args.cpu, 'num_confs': args.num_confs,
            'versions': versions(), 'code_hashes': code_hashes()}


def check_gate(report_path, expected_protocol, seed):
    if not report_path:
        raise ValueError('Candidate docking requires --validation-report from a passed reference run')
    report = read_json(report_path)
    if report.get('stage') != 'reference' or not report.get('gate_passed'):
        raise ValueError('Reference validation has not passed; candidate ranking remains blocked')
    if report['protocol'] != expected_protocol or seed not in report['vina_seeds']:
        raise ValueError('Reference validation does not cover these inputs or docking parameters')
    return report


def input_audit(entries, workdir):
    states = [state_record(entry)[1] for entry in entries]
    write_json(workdir / 'input_audit.json', states)
    summary = {'total_candidates': len(entries), 'state_status': dict(Counter(r['status'] for r in states)),
               'scope_counts': dict(Counter(r.get('scope', 'unspecified') for r in entries)),
               'requires_state_review_hint': sum(bool(r.get('state_review_hints')) for r in states),
               'protonation_and_tautomer_coverage': 'not predicted; preserve input states',
               'meaning': 'pending means eligible for 3D preparation, not docking validated'}
    write_json(workdir / 'input_audit_summary.json', summary)
    return summary


def run_candidates(score, entries, receptors, workdir, args):
    current_protocol = protocol(score, receptors, args)
    check_gate(args.validation_report, current_protocol, args.seed)
    results = {}

    def one(entry):
        label = entry['candidate_id']
        folder = workdir / 'candidates' / hashlib.sha256(label.encode()).hexdigest()[:24]
        folder.mkdir(parents=True, exist_ok=True)
        record = {**entry, 'status': 'running'}
        write_json(folder / 'result.json', record)
        try:
            scored = score.score_one(label, entry['smiles'], receptors, folder,
                                     preparation_seed=args.preparation_seed, num_confs=args.num_confs,
                                     resume=args.resume, vina=args.vina, exhaustiveness=args.exhaustiveness,
                                     num_modes=args.num_modes, seed=args.seed, cpu=args.cpu, timeout=args.timeout)
            record.update(status='ok', scores=asdict(scored))
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder / 'result.json', record)
        return record

    def checkpoint():
        ordered = [results[r['candidate_id']] for r in entries if r['candidate_id'] in results]
        write_json(workdir / 'results.json', ordered)
        table = io.StringIO(newline='')
        writer = csv.writer(table, delimiter='\t', lineterminator='\n')
        writer.writerow(score.HEADER)
        for row in sorted((r for r in ordered if r['status'] == 'ok'), key=lambda r: -r['scores']['combined']):
            writer.writerow(score.Scored(**row['scores']).row())
        dest = workdir / 'subtype_scores.tsv'
        temp = dest.with_suffix('.tsv.tmp')
        temp.write_text(table.getvalue(), encoding='utf-8')
        replace_file(temp, dest)
        write_json(workdir / 'summary.json', {'expected': len(entries), 'completed': len(ordered),
                   'status_counts': dict(Counter(r['status'] for r in ordered)),
                   'combined_score': 'Original formula and weights; exploratory only'})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, entry) for entry in entries]
        for future in as_completed(futures):
            row = future.result()
            results[row['candidate_id']] = row
            checkpoint()
            print(f"{len(results)}/{len(entries)} {row['candidate_id']} {row['status']}", flush=True)
    checkpoint()
    if args.out:
        destination = Path(args.out).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((workdir / 'subtype_scores.tsv').read_bytes())
    return 0 if all(r['status'] == 'ok' for r in results.values()) else 2


def main(score):
    parser = argparse.ArgumentParser(description='Original dual-target scoring with audited preparation and reference validation')
    parser.add_argument('--stage', choices=['audit', 'reference', 'candidates'], default='reference')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--smiles-column', default='smiles')
    parser.add_argument('--label-column')
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--receptor-dir', type=Path, default=BASE / 'outputs/receptors_v3')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--vina')
    parser.add_argument('--exhaustiveness', type=int, default=score.EXHAUSTIVENESS)
    parser.add_argument('--num-modes', type=int, default=score.NUM_MODES)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--num-confs', type=int, default=20)
    parser.add_argument('--preparation-seed', type=int, default=20261002)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--reference-repeats', type=int, default=5)
    parser.add_argument('--vina-seeds', type=int, nargs='+', default=[20261002, 20261003, 20261004])
    parser.add_argument('--validation-report', type=Path)
    parser.add_argument('--out', type=Path, help='Optional final TSV copy; checkpoints remain in workdir')
    args = parser.parse_args()
    if min(args.exhaustiveness, args.num_modes, args.cpu, args.workers, args.timeout,
           args.num_confs, args.reference_repeats) <= 0 or min(args.seed, args.preparation_seed, *args.vina_seeds) < 0:
        parser.error('Sampling/resource values must be positive; random seeds nonnegative')
    if len(set(args.vina_seeds)) != len(args.vina_seeds):
        parser.error('Reference Vina seeds must be distinct')
    if args.stage in ('audit', 'candidates') and args.input is None:
        parser.error('--input is required for audit/candidates')
    if args.stage == 'reference' and args.input:
        parser.error('Reference validation uses bound reference fixtures, not --input')
    if args.stage == 'candidates' and (not args.validation_report or not read_json(args.validation_report).get('gate_passed')):
        parser.error('Candidate docking requires a passed --validation-report')
    args.workdir = (args.workdir or BASE / 'outputs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.stage)).resolve()
    args.receptor_dir = args.receptor_dir.resolve()
    entries = read_entries(args.input, args.smiles_column, args.label_column) if args.input else []
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('resume', 'out')}
    signature = {'stage': args.stage, 'settings': settings, 'code_hashes': code_hashes(),
                 'versions': versions(), 'python': sys.executable,
                 'input_sha256': sha256(args.input) if args.input else None,
                 'validation_report_sha256': sha256(args.validation_report) if args.validation_report else None,
                 'receptor_sources': {n: sha256(c['pdb']) for n, c in score.RECEPTORS.items()}}
    if args.stage != 'audit':
        args.vina = str(resolve_vina(args.vina))
        signature['vina_sha256'] = sha256(args.vina)
    if args.stage == 'reference':
        signature['reference_sha256'] = {p.name: sha256(p) for p in sorted((BASE / 'data/reference_validation').glob('*.sdf'))}
    work_manifest(args.workdir, signature, args.resume)
    if entries:
        write_json(args.workdir / 'input_snapshot.json', entries)
        summary = input_audit(entries, args.workdir)
        if args.stage == 'audit':
            print(summary, flush=True)
            return 0
    print(f'Workdir: {args.workdir}', flush=True)
    print(f'CPU budget: {args.workers} jobs x {args.cpu} threads; logical CPUs={os.cpu_count()}', flush=True)
    receptors = score.prepare_receptors(args.receptor_dir, resume=True)
    if args.stage == 'reference':
        from .validation import run_reference
        return run_reference(score, receptors, args.workdir, args)
    return run_candidates(score, entries, receptors, args.workdir, args)
