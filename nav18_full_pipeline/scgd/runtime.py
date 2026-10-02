"""Portable, recorded subprocesses and hash-checked resumable docking jobs."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    replace_file(temporary, path)


def replace_file(temporary, destination):
    """Windows readers/indexers may briefly hold a destination without delete sharing."""
    for attempt in range(11):
        try:
            Path(temporary).replace(destination)
            return
        except PermissionError:
            if attempt == 10:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 0.5))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def versions():
    return {p: version(p) for p in ('rdkit', 'meeko', 'gemmi', 'pdbfixer', 'openmm', 'numpy')}


def code_hashes():
    paths = [BASE / 'score.py', *sorted((BASE / 'scgd').glob('*.py'))]
    return {str(p.relative_to(BASE)): sha256(p) for p in paths}


def resolve_vina(value=None):
    candidates = ([value] if value else [os.environ.get('VINA_EXECUTABLE'), shutil.which('vina'),
                  BASE.parent / 'nav1p8-molecule-generator/vina_1.2.7_win.exe',
                  '/opt/homebrew/bin/vina'])
    for candidate in candidates:
        if candidate:
            path = Path(shutil.which(str(candidate)) or candidate).expanduser().resolve()
            if path.is_file():
                return path
    raise FileNotFoundError('AutoDock Vina not found; pass --vina or set VINA_EXECUTABLE')


def logged_run(command, log_path, timeout=900):
    command = list(map(str, command))
    start = time.monotonic()
    log = {'command': command, 'python': sys.executable}
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=timeout)
        log.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   status='ok' if result.returncode == 0 else 'failed')
    except subprocess.TimeoutExpired as exc:
        log.update(status='timeout', stdout=str(exc.stdout or ''), stderr=str(exc.stderr or ''))
        raise
    except OSError as exc:
        log.update(status='launch_failed', error=str(exc))
        raise
    finally:
        log['elapsed_seconds'] = time.monotonic() - start
        write_json(log_path, log)
    if result.returncode:
        raise RuntimeError(f'Command failed; see {log_path}: {result.stderr[-1200:]}')
    return result


def work_manifest(directory, signature, resume=False):
    directory = Path(directory)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        if not resume:
            raise FileExistsError(f'Run exists; use --resume: {directory}')
        if read_json(manifest) != signature:
            raise ValueError('Resume inputs, code, dependencies or parameters changed; use a new workdir')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Refusing to overwrite a nonempty directory: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        write_json(manifest, signature)


def dock_job(ligand, receptor, pose, center, size, *, vina=None, exhaustiveness=8,
             num_modes=9, seed=20261002, cpu=2, timeout=600, resume=False):
    """Reuse only an intact successful job with identical input hashes and parameters."""
    ligand, receptor, pose = map(Path, (ligand, receptor, pose))
    executable = resolve_vina(vina)
    if min(exhaustiveness, num_modes, cpu, timeout) <= 0 or seed < 0:
        raise ValueError('Invalid Vina resource or sampling parameters')
    signature = {'ligand_sha256': sha256(ligand), 'receptor_sha256': sha256(receptor),
                 'vina_sha256': sha256(executable), 'center': list(center), 'size': float(size),
                 'exhaustiveness': exhaustiveness, 'num_modes': num_modes, 'seed': seed, 'cpu': cpu,
                 'timeout': timeout}
    record = pose.with_suffix('.job.json')
    if record.exists():
        previous = read_json(record)
        if not resume:
            raise FileExistsError(f'Job exists: {record}')
        if previous['signature'] != signature:
            raise ValueError(f'Resume job signature changed: {record}')
        if previous['status'] == 'ok':
            if not pose.exists() or sha256(pose) != previous['pose_sha256']:
                raise ValueError(f'Completed pose is missing or changed: {pose}')
            return pose
    pose.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted/failed attempt must never leave a stale pose for parsing.
    if pose.exists():
        if not resume:
            raise FileExistsError(f'Pose exists without a reusable job: {pose}')
        pose.unlink()
    state = {'signature': signature, 'status': 'running'}
    write_json(record, state)
    command = [executable, '--receptor', receptor, '--ligand', ligand, '--out', pose,
               '--exhaustiveness', exhaustiveness, '--num_modes', num_modes, '--seed', seed, '--cpu', cpu]
    for axis, value in zip('xyz', center):
        command += [f'--center_{axis}', value, f'--size_{axis}', size]
    try:
        logged_run(command, pose.with_suffix('.log.json'), timeout)
        scores = [float(s) for s in re.findall(r'REMARK VINA RESULT:\s+(-?[\d.]+)', pose.read_text())]
        if not scores or not all(__import__('math').isfinite(s) for s in scores):
            raise ValueError('Vina did not return finite pose scores')
        state.update(status='ok', scores=scores, pose_sha256=sha256(pose))
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(record, state)
    return pose
