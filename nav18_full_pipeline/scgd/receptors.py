"""PDBFixer repair and Meeko residue-template typing; no per-atom sphere cuts.

Adapted from the main project's scripts/prepare_docking_validation.py.
Experimental atom coordinates and residue identifiers must remain unchanged.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from openmm import Platform
from openmm.app import PDBFile
from pdbfixer import PDBFixer

from .geometry import pdb_atoms, select_chain
from .runtime import logged_run, read_json, sha256, versions, write_json


def prepare_receptor(source, directory, chain, center, size=24.0, resume=False):
    source, directory = Path(source), Path(directory)
    signature = {'source_sha256': sha256(source), 'chain': chain, 'center': list(center),
                 'size': float(size), 'versions': versions(), 'preparer_sha256': sha256(__file__),
                 'geometry_sha256': sha256(Path(__file__).with_name('geometry.py'))}
    audit_path = directory / 'audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        if not resume or audit['signature'] != signature:
            raise ValueError('Receptor directory exists or preparation inputs changed; use a fresh directory')
        if audit.get('status') == 'ok':
            for name, digest in audit['files'].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f'Receptor artifact changed: {name}')
            return directory / 'receptor.pdbqt'
    directory.mkdir(parents=True, exist_ok=True)
    audit = {'signature': signature, 'status': 'running', 'missing_loops_built': False,
             'limitations': 'Rigid protein-only; lipids/waters excluded; Meeko template protonation; no pKa prediction.'}
    write_json(audit_path, audit)
    try:
        selected = directory / 'selected.pdb'
        select_chain(source, selected, chain)
        before = pdb_atoms(selected)
        platform = Platform.getPlatformByName('CPU')
        platform.setPropertyDefaultValue('Threads', '1')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        fixer = PDBFixer(filename=str(selected), platform=platform)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        audit['missing_heavy_atoms_repaired'] = [
            {'chain': residue.chain.id, 'residue': residue.id, 'name': residue.name,
             'missing_atoms': [a.name for a in atoms]}
            for residue, atoms in fixer.missingAtoms.items()]
        fixer.addMissingAtoms(seed=20260930)
        repaired = directory / 'repaired.pdb'
        with repaired.open('w') as stream:
            PDBFile.writeFile(fixer.topology, fixer.positions, stream, keepIds=True)
        after = pdb_atoms(repaired)
        if not set(before).issubset(after):
            raise ValueError('Repair removed observed atoms or changed identifiers')
        displacement = max(float(np.linalg.norm(after[k] - p)) for k, p in before.items())
        if displacement > 0.002:
            raise ValueError(f'Repair moved observed atoms by {displacement} A')
        audit.update(observed_heavy_atoms=len(before), repaired_heavy_atoms=len(after),
                     max_observed_atom_displacement_A=displacement)
        exclusions = {}
        for attempt in range(4):
            active = repaired
            if exclusions:
                filtered = directory / 'explicitly_filtered.pdb'
                filtered.write_text('\n'.join(l for l in repaired.read_text().splitlines()
                    if not l.startswith('ATOM  ') or f'{l[21]}:{l[22:27].strip()}' not in exclusions) + '\n')
                active = directory / 'meeko_input.pdb'
                select_chain(filtered, active, chain)
            command = [sys.executable, '-m', 'meeko.cli.mk_prepare_receptor', '--read_pdb', active,
                       '-p', directory / 'receptor.pdbqt', '--write_pdb', directory / 'prepared.pdb',
                       '-j', directory / 'receptor.json', '--default_altloc', 'A']
            log = directory / f'preparation_log_{attempt}.json'
            try:
                logged_run(command, log)
                break
            except RuntimeError:
                message = read_json(log)['stderr']
                bad = (set(re.findall(r"No template matched for residue_key='([^']+)'", message)) |
                       set(re.findall(r'corresponding to key ([A-Za-z0-9:]+)', message))) - set(exclusions)
                if not bad or attempt == 3:
                    raise
                for key in sorted(bad):
                    ch, residue = key.split(':')
                    coords = np.array([p for k, p in after.items() if k[0] == ch and k[1] == residue])
                    if not len(coords):
                        raise ValueError(f'Cannot audit failed residue {key}')
                    distance = float(np.linalg.norm(np.maximum(np.abs(coords - center) - size / 2, 0), axis=1).min())
                    if distance <= 8:
                        raise ValueError(f'Meeko template failure near search box: {key}; manual repair required')
                    exclusions[key] = {'min_distance_to_box_A': distance,
                                       'reason': 'Meeko template/bond mismatch; all residue atoms >8 A outside box'}
        audit['explicit_distant_template_exclusions'] = exclusions
        # Include exactly the heavy atoms present in the prepared receptor in contact checks.
        pdbqt = directory / 'receptor.pdbqt'
        lines = [l for l in pdbqt.read_text().splitlines() if l.startswith(('ATOM  ', 'HETATM'))]
        if not lines or any(not np.isfinite(float(l[70:76])) for l in lines):
            raise ValueError('Empty receptor or nonfinite receptor charges')
        ntypes = Counter(l.split()[-1] for l in lines if l[12:16].strip() == 'N')
        if not ntypes or ntypes.get('NA', 0):
            raise ValueError(f'Invalid protein backbone nitrogen typing: {ntypes}')
        audit['backbone_nitrogen_types'] = dict(ntypes)
        write_json(directory / 'box.json', {'center': list(center), 'size': [size] * 3})
        audit.update(status='ok', files={name: sha256(directory / name) for name in
                     ('selected.pdb', 'repaired.pdb', 'prepared.pdb', 'receptor.pdbqt', 'receptor.json', 'box.json')})
    except Exception as exc:
        audit.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(audit_path, audit)
    return directory / 'receptor.pdbqt'
