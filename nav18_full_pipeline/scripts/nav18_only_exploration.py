"""Frozen Nav1.8-only exploration: independent reference -> pilot -> library.

This does not alter or release scgd's original dual-target production gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import io
import os
from pathlib import Path
import shutil
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from scgd.ligand_quality import state_record
from scgd.prep import prep_ligand
from scgd.runtime import (code_hashes, dock_job, read_json, replace_file,
                          resolve_vina, sha256, versions, work_manifest, write_json)
from scgd.validation import assess_pose, reference_molecule
from scgd.workflow import input_audit, read_entries
from nav18_pose_quality import assess_candidate_pose

MODE = 'nav18_only_exploration'
CENTER = (137.204, 136.509, 127.538)
SIZE = 24.0
PREP_SEED = 20261012
SEEDS = (20261012, 20261013, 20261014)
REPEATS = 5
PARENT_ID = 'dce16c0c1568b893'
INPUT = BASE.parent / 'nav1p8-molecule-generator/outputs/a803467_expanded_20261001/all_retained.csv'
SELECTION = BASE.parent / 'nav1p8-molecule-generator/results/a803467_20261001/library/pilot_selection.json'
SOURCES = {
    'receptor.pdbqt': BASE / 'outputs/nav18_lipid_control_20261002/zwitterion/receptor_with_lipid.pdbqt',
    'protein.pdb': BASE / 'outputs/receptors_20261002/nav18/prepared.pdb',
    'lipid.sdf': BASE / 'outputs/nav18_lipid_control_20261002/zwitterion/truncated_lpe.sdf',
    'lipid_audit.json': BASE / 'outputs/nav18_lipid_control_20261002/zwitterion/preparation_audit.json',
    'reference.sdf': BASE / 'data/reference_validation/nav18.sdf',
    'deposited.pdb': BASE / 'receptors/7WE4.pdb',
}
# Chosen without docking scores from the pre-existing diversity panel. Frozen before running.
PILOT_IDS = ('c5f6502e9184ecce', '6f286ae3198cc668', '8b59b199f37d9e78', '017c979568cdc3ff')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def unknown_fields():
    return dict(mode=MODE, nav15_risk='unknown', nav15_score=None, delta=None,
                combined_score=None, production_gate_released=False)


def protocol():
    return dict(mode=MODE, files={name: sha256(path) for name, path in SOURCES.items()},
                center=list(CENTER), size=SIZE, exhaustiveness=8, num_modes=9, cpu=2,
                timeout=900, num_confs=20, preparation_policy='lowest_energy_valid_input_state_conformer',
                versions=versions(), vina_sha256=sha256(resolve_vina()),
                production_code_hashes=code_hashes(),
                exploration_code_hashes={p.name: sha256(p) for p in
                    (Path(__file__), Path(__file__).with_name('nav18_pose_quality.py'))})


def snapshot(workdir, signature):
    folder = workdir / 'frozen_inputs'
    folder.mkdir(exist_ok=True)
    for name, source in SOURCES.items():
        target = folder / name
        if not target.exists():
            shutil.copy2(source, target)
        if sha256(target) != signature['protocol']['files'][name]:
            raise ValueError(f'Frozen artifact changed: {target}')
    return folder


@contextmanager
def exclusive_run(workdir):
    """An OS-released lock permits safe restart after interruption, without stale PID guessing."""
    import msvcrt
    path = workdir / '.run.lock'
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another process is using this run directory') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def recorded_dock(ligand, pose, frozen, seed, resume):
    # Preserve failed artifacts before an explicitly resumed same-protocol retry.
    job = pose.with_suffix('.job.json')
    if resume and job.exists() and read_json(job).get('status') != 'ok':
        archive = pose.parent / ('attempt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        archive.mkdir()
        for path in (pose, job, pose.with_suffix('.log.json'), pose.parent / 'result.json'):
            if path.exists():
                shutil.copy2(path, archive / path.name)
    return dock_job(ligand, frozen / 'receptor.pdbqt', pose, CENTER, SIZE,
                    exhaustiveness=8, num_modes=9, seed=seed, cpu=2, timeout=900, resume=resume)


def gate_summary(rows, current_protocol, evidence):
    good = [r for r in rows if r['status'] == 'ok']
    passed = sum(r.get('reference_passed', False) for r in good)
    complete = len(rows) == REPEATS * len(SEEDS) and len(good) == len(rows)
    return dict(stage='independent_nav18_reference', **unknown_fields(), protocol=current_protocol,
                reference_repeats=REPEATS, vina_seeds=list(SEEDS), preparation_seed=PREP_SEED,
                expected=REPEATS * len(SEEDS), completed=len(rows), successful=len(good),
                top1_pass_count=passed,
                nav18_exploration_eligible=complete and passed / (REPEATS * len(SEEDS)) >= 0.8,
                median_top1_rmsd_A=float(np.median([r['top1_rmsd_A'] for r in good])) if good else None,
                criteria='top1 reference RMSD <=2 A; native protein contact recall >=0.5; '
                         'no protein/lipid heavy pair <2 A; candidate pose geometry QA pass; >=12/15 passes',
                evidence=evidence,
                sampling='5 fresh independently embedded input pools x 3 new Vina seeds; '
                         'input pools are not asserted to be 5 distinct conformational basins',
                interpretation='Pose validation for fixed A-803467 Site C exploration only; '
                               'not activity, selectivity or a validated candidate affinity ranking')


def check_reference_gate(path, current_protocol, seed):
    report = read_json(path)
    if (report.get('stage') != 'independent_nav18_reference'
            or not report.get('nav18_exploration_eligible')
            or report.get('production_gate_released') is not False):
        raise ValueError('Independent Nav1.8 exploration validation has not passed')
    if report['protocol'] != current_protocol or seed not in report['vina_seeds']:
        raise ValueError('Reference does not cover the current frozen protocol/seed')
    if report.get('successful') != 15 or report.get('top1_pass_count', 0) < 12 or not report.get('evidence'):
        raise ValueError('Incomplete reference evidence')
    for relative, digest in report['evidence'].items():
        if sha256(path.parent / relative) != digest:
            raise ValueError(f'Reference evidence changed: {relative}')
    if 'results.json' not in report['evidence']:
        raise ValueError('Missing reference results evidence')
    rows = read_json(path.parent / 'results.json')
    expected = {(repeat, seed) for repeat in range(REPEATS) for seed in SEEDS}
    actual = {(r['repeat'], r['vina_seed']) for r in rows}
    if len(rows) != 15 or actual != expected or any(r['status'] != 'ok' for r in rows):
        raise ValueError('Reference results are incomplete or duplicated')
    passes = sum(r.get('reference_passed', False) for r in rows)
    if passes != report['top1_pass_count'] or passes < 12:
        raise ValueError('Reference pass count disagrees with evidence')
    for row in rows:
        relative = str(Path('jobs') / f"c{row['repeat']}_s{row['vina_seed']}" / 'result.json')
        if relative not in report['evidence'] or read_json(path.parent / relative) != row:
            raise ValueError('Reference job evidence is missing or inconsistent')
    return report


def run_reference(workdir, frozen, current_protocol, args):
    import score
    # Validate the fixture against deposited experimental coordinates, then discard all coordinates for prep.
    reference = reference_molecule('nav18', score)
    smiles = Chem.MolToSmiles(reference)
    folder = workdir / 'reference'
    folder.mkdir(exist_ok=True)
    jobs, rows = [], []
    for repeat in range(REPEATS):
        ligand = folder / 'ligands' / f'input_{repeat}.pdbqt'
        prep_ligand(smiles, ligand, seed=PREP_SEED + repeat, candidate_id=PARENT_ID,
                    num_confs=20, resume=args.resume)
        jobs.extend((repeat, seed, ligand) for seed in SEEDS)
    # Describe conformer diversity honestly; independent pools may converge to the same minimum.
    mols = [Chem.SDMolSupplier(str(folder / 'ligands' / f'input_{i}.sdf'), removeHs=True)[0]
            for i in range(REPEATS)]
    pairs = [{'i': i, 'j': j, 'fitted_heavy_rmsd_A': rdMolAlign.GetBestRMS(Chem.Mol(mols[i]), mols[j])}
             for i in range(REPEATS) for j in range(i)]
    write_json(folder / 'input_conformer_diversity.json', pairs)

    def one(job):
        repeat, seed, ligand = job
        run = folder / 'jobs' / f'c{repeat}_s{seed}'
        run.mkdir(parents=True, exist_ok=True)
        row = dict(repeat=repeat, vina_seed=seed, status='running')
        try:
            pose = recorded_dock(ligand, run / 'poses.pdbqt', frozen, seed, args.resume)
            qa = assess_candidate_pose(pose, smiles, frozen / 'protein.pdb', frozen / 'lipid.sdf', CENTER, SIZE)
            assessment = assess_pose(pose, reference, frozen / 'protein.pdb')
            row.update(assessment, quality=qa, status='ok',
                       reference_passed=assessment['top1_passed'] and qa['quality_status'] == 'pass')
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(run / 'result.json', row)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            write_json(folder / 'results.json', sorted(rows, key=lambda r: (r['repeat'], r['vina_seed'])))
            write_json(folder / 'report.json', gate_summary(rows, current_protocol, {}))
            print(f"reference {len(rows)}/15 {row['status']} pass={row.get('reference_passed')}", flush=True)
    evidence = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*')
                if p.is_file() and p.name not in ('report.json',) and 'attempt_' not in str(p)}
    report = gate_summary(rows, current_protocol, evidence)
    write_json(folder / 'report.json', report)
    check_reference_gate(folder / 'report.json', current_protocol, SEEDS[0])
    return reference


def candidate_folder(workdir, label):
    return workdir / 'candidates' / hashlib.sha256(label.encode()).hexdigest()[:24]


def run_one(entry, workdir, frozen, resume):
    folder = candidate_folder(workdir, entry['candidate_id'])
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / 'result.json'
    if result.exists() and resume:
        previous = read_json(result)
        if previous.get('docking_status') in ('ok', 'deferred'):
            if previous.get('smiles') != entry['smiles'] or previous.get('candidate_id') != entry['candidate_id']:
                raise ValueError('Candidate identity changed')
            if any(previous.get(key) != value for key, value in unknown_fields().items()):
                raise ValueError('Candidate result mode/unknown-state contract changed')
            if previous['docking_status'] == 'ok' and not previous.get('artifacts'):
                raise ValueError('Completed candidate lacks artifact hashes')
            for name, digest in previous.get('artifacts', {}).items():
                if sha256(folder / name) != digest:
                    raise ValueError(f'Candidate artifact changed: {folder / name}')
            return previous
        archive = folder / ('attempt_result_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        archive.mkdir()
        for path in (result, folder / 'ligand.preparation.json'):
            if path.exists():
                shutil.copy2(path, archive / path.name)
    _, state = state_record(entry)
    row = dict(entry, **unknown_fields(), nav18_score=None, docking_status='running',
               quality_status='not_assessed', quality_flags=[], state_status=state['status'],
               state_review_hints=state.get('state_review_hints', []),
               state_policy='input_state_preserved; protonation/tautomers not enumerated',
               pose_path=None, started_at=utcnow())
    start = time.monotonic()
    write_json(result, row)
    try:
        if state['status'] != 'pending':
            row.update(docking_status='deferred', defer_reason=state['status'],
                       unspecified_stereo=state.get('unspecified_stereo', []))
        else:
            ligand = prep_ligand(entry['smiles'], folder / 'ligand.pdbqt', seed=PREP_SEED,
                                 candidate_id=entry['candidate_id'], num_confs=20, resume=resume)
            pose = recorded_dock(ligand, folder / 'poses.pdbqt', frozen, SEEDS[0], resume)
            qa = assess_candidate_pose(pose, entry['smiles'], frozen / 'protein.pdb',
                                       frozen / 'lipid.sdf', CENTER, SIZE)
            row.update(docking_status='ok', nav18_score=qa['affinity_kcal_mol'],
                       quality_status=qa['quality_status'], quality_flags=qa['quality_flags'],
                       pose_quality=qa, pose_path=str(pose.relative_to(workdir)),
                       artifacts={p.name: sha256(p) for p in folder.iterdir()
                                  if p.is_file() and p.name != 'result.json'})
    except Exception as exc:
        row.update(docking_status='failed', error=f'{type(exc).__name__}: {exc}')
    row.update(elapsed_seconds=time.monotonic() - start, finished_at=utcnow())
    write_json(result, row)
    return row


def write_table(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row if k not in ('artifacts', 'pose_quality')))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    import json
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                         else v for k, v in row.items() if k in columns})
    temp = path.with_suffix('.csv.tmp')
    temp.write_text(output.getvalue(), encoding='utf-8-sig')
    replace_file(temp, path)


def checkpoint(workdir, entries, results, stage):
    rows = [results.get(e['candidate_id'], dict(e, **unknown_fields(), nav18_score=None,
                     docking_status='queued', quality_status='not_assessed')) for e in entries]
    write_json(workdir / f'{stage}_results.json', rows)
    write_table(workdir / f'{stage}_status.csv', rows)
    complete = all(r['docking_status'] not in ('queued', 'running') for r in rows)
    summary = dict(stage=stage, **unknown_fields(), expected=len(entries),
                   completed=len(results), complete=complete, updated_at=utcnow(),
                   status_counts=dict(Counter(r['docking_status'] for r in rows)),
                   quality_counts=dict(Counter(r['quality_status'] for r in rows)),
                   null_columns_in_csv='Empty cells represent null/unknown, never zero',
                   scope='Rigid protein + observed truncated fixed lipid, A-803467 Site C, input chemical state')
    summary['severe_quality_count'] = sum(bool(r.get('pose_quality', {}).get('severe_quality_flags')) for r in rows)
    write_json(workdir / f'{stage}_summary.json', summary)
    if complete:
        eligible = sorted((r for r in rows if r['docking_status'] == 'ok' and r['quality_status'] == 'pass'),
                          key=lambda r: (r['nav18_score'], r['candidate_id']))
        if eligible:
            write_table(workdir / f'{stage}_raw_score_order.csv', eligible)
    return summary


def run_batch(entries, workdir, frozen, current_protocol, args, stage):
    check_reference_gate(workdir / 'reference/report.json', current_protocol, SEEDS[0])
    results = {}
    checkpoint(workdir, entries, results, stage)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, e, workdir, frozen, args.resume or stage == 'library') for e in entries]
        for future in as_completed(futures):
            row = future.result()
            results[row['candidate_id']] = row
            if len(results) % 10 == 0 or len(results) == len(entries):
                checkpoint(workdir, entries, results, stage)
            print(f"{stage} {len(results)}/{len(entries)} {row['candidate_id']} "
                  f"{row['docking_status']} {row['quality_status']} score={row.get('nav18_score')}", flush=True)
    return checkpoint(workdir, entries, results, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, default=BASE / 'outputs/nav18_only_20261002')
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=('all', 'reference', 'pilot', 'library'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error('workers must be 1..8 (two Vina CPUs per job)')
    workdir = args.workdir.resolve()
    entries = read_entries(args.input)
    current_protocol = protocol()
    signature = dict(protocol=current_protocol, input_sha256=sha256(args.input),
                     selection_sha256=sha256(SELECTION), pilot_ids=list(PILOT_IDS),
                     preparation_seed=PREP_SEED, vina_seeds=list(SEEDS), reference_repeats=REPEATS)
    work_manifest(workdir, signature, args.resume)
    with exclusive_run(workdir):
        frozen = snapshot(workdir, signature)
        input_audit(entries, workdir)
        parent = read_json(SELECTION)['candidates'][0]
        if parent['candidate_id'] != PARENT_ID:
            raise ValueError('Unexpected parent in historical selection')
        lookup = {e['candidate_id']: e for e in entries}
        pilot = [parent, *(lookup[label] for label in PILOT_IDS)]
        write_json(workdir / 'pilot_selection.json', dict(policy='Pre-score, pre-existing reviewed panel; '
                   'site enumeration strict 3-site edit, Mol2Mol strict acetylpiperazine, '
                   'amide recombination expanded fused indole, amide recombination outside-series '
                   'amide-N/aryl CH2 insertion with 9 rotatable bonds. '
                   'Technical pilot covering three methods and three scopes, not an accuracy benchmark.', entries=pilot))
        write_json(workdir / 'run_state.json', dict(status='running', stage=args.stage, pid=os.getpid(), updated_at=utcnow()))
        try:
            if args.stage in ('all', 'reference'):
                run_reference(workdir, frozen, current_protocol, args)
            if args.stage in ('all', 'pilot'):
                summary = run_batch(pilot, workdir, frozen, current_protocol, args, 'pilot')
                if summary['status_counts'].get('ok') != len(pilot) or summary['severe_quality_count']:
                    raise ValueError('Pilot technical failures or severe pose flags require review before library execution')
            if args.stage in ('all', 'library'):
                # Even direct --stage library requires the recorded technical pilot.
                pilot_summary = read_json(workdir / 'pilot_summary.json')
                if (not pilot_summary.get('complete') or pilot_summary['status_counts'].get('ok') != 5
                        or pilot_summary.get('severe_quality_count', 0)):
                    raise ValueError('Complete parent + four pilot required')
                for entry in pilot:
                    record = read_json(candidate_folder(workdir, entry['candidate_id']) / 'result.json')
                    for name, digest in record.get('artifacts', {}).items():
                        if sha256(candidate_folder(workdir, entry['candidate_id']) / name) != digest:
                            raise ValueError('Pilot evidence changed')
                summary = run_batch(entries, workdir, frozen, current_protocol, args, 'library')
                if summary['status_counts'].get('failed', 0):
                    write_json(workdir / 'run_state.json', dict(status='finished_with_failures',
                               stage=args.stage, updated_at=utcnow()))
                    return 2
            write_json(workdir / 'run_state.json', dict(status='finished', stage=args.stage, updated_at=utcnow()))
        except BaseException as exc:
            write_json(workdir / 'run_state.json', dict(status='stopped', stage=args.stage,
                       error=f'{type(exc).__name__}: {exc}', updated_at=utcnow()))
            raise


if __name__ == '__main__':
    sys.exit(main())
