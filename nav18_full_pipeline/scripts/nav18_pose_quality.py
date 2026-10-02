"""Geometry/identity audit for Nav1.8-only docking, without an activity score.

The parent reference's experimental RMSD/contact criteria deliberately do not
apply to candidates. ``quality_status`` describes the first (Vina-ranked) mode;
all other modes are audited and retained without picking a better-looking mode.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgd.geometry import fixed_frame_rmsd, pdb_atoms
from scgd.ligand_quality import geometry_issues
from scgd.runtime import replace_file


DEFAULT_CENTER = (137.204, 136.509, 127.538)
DEFAULT_SIZE = 24.0


def _cache_key(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=8)
def _protein_data(path, _size, _mtime):
    # Exactly the ATOM/heavy-atom/residue policy in scgd.geometry.pose_contacts.
    atoms = pdb_atoms(path)
    xyz = np.asarray(list(atoms.values()), dtype=float)
    if not atoms or not np.isfinite(xyz).all():
        raise ValueError('Protein has no finite heavy-atom coordinates')
    xyz.setflags(write=False)
    return tuple(atoms), xyz


@lru_cache(maxsize=8)
def _lipid_data(path, _size, _mtime):
    mols = list(Chem.SDMolSupplier(path, removeHs=True))
    if len(mols) != 1 or mols[0] is None or not mols[0].GetNumAtoms():
        raise ValueError('Lipid must contain exactly one valid molecule')
    xyz = np.asarray(mols[0].GetConformer().GetPositions(), dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError('Lipid has nonfinite coordinates')
    xyz.setflags(write=False)
    return xyz


def _canonical(mol, stereo=True):
    copy = Chem.RemoveHs(Chem.Mol(mol))
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    if not stereo:
        Chem.RemoveStereochemistry(copy)
    # isomericSmiles=True retains isotope information even for graph-only checks.
    return Chem.MolToSmiles(copy, isomericSmiles=True)


def _protein_contacts(xyz, keys, protein):
    # Bounding-box filtering preserves all <=4 A contacts and <2 A clashes.
    # If its minimum is >4 A, use the full receptor to obtain an exact minimum.
    near = np.all((protein >= xyz.min(axis=0) - 4.) &
                  (protein <= xyz.max(axis=0) + 4.), axis=1)
    indices = np.flatnonzero(near)
    distances = np.linalg.norm(xyz[:, None, :] - protein[near][None, :, :], axis=2)
    if not indices.size or distances.min() > 4.:
        indices = np.arange(len(protein))
        distances = np.linalg.norm(xyz[:, None, :] - protein[None, :, :], axis=2)
    residues = sorted({':'.join(keys[i][:3]) for i, d in
                       zip(indices, distances.min(axis=0)) if d <= 4.})
    return {'contact_residues_4A': residues,
            'min_protein_heavy_distance_A': float(distances.min()),
            'heavy_pairs_below_2A': int((distances < 2.).sum())}


def _environment_metrics(mol, protein_data, lipid, center, size):
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    distances = np.linalg.norm(xyz[:, None, :] - lipid[None, :, :], axis=2)
    # Requested physical box, not an undocumented Vina grid-rounding estimate.
    margin = float(np.min(size / 2. - np.abs(xyz - center)))
    return {**_protein_contacts(xyz, *protein_data),
            'min_lipid_heavy_distance_A': float(distances.min()),
            'lipid_heavy_pairs_below_2A': int((distances < 2.).sum()),
            'min_requested_box_edge_margin_A': margin,
            'outside_requested_box': margin < 0.,
            'within_1A_of_requested_box_edge': margin < 1.}


def assess_candidate_pose(pose_path, smiles, receptor_pdb, lipid_sdf,
                          center=DEFAULT_CENTER, size=DEFAULT_SIZE):
    """Export and audit all Vina modes; return only JSON-serializable values.

    Invalid/missing scores, coordinates, or chemical identity raise ValueError.
    Unspecified input stereochemistry is deferred, never assigned by docking.
    ``quality_flags`` and flattened environment fields refer to mode 1.
    ``modes`` holds the corresponding fields for every exported mode.
    ``all_pose_rmsd_to_top1_A`` is descriptive fixed-frame within-molecule
    consistency; it is not a comparison to A-803467 or an acceptance criterion.
    """
    expected = Chem.MolFromSmiles(smiles)
    if expected is None or not expected.GetNumAtoms():
        raise ValueError('Invalid input SMILES')
    if len(Chem.GetMolFrags(expected)) != 1 or any(a.GetAtomicNum() == 0 for a in expected.GetAtoms()):
        raise ValueError('Input requires component review')
    unresolved = [f'{info.type}:{info.centeredOn}' for info in Chem.FindPotentialStereo(expected)
                  if info.specified != Chem.StereoSpecified.Specified]
    if unresolved:
        raise ValueError('Input requires stereochemistry review: ' + ', '.join(unresolved))
    expected_smiles = _canonical(expected)
    expected_graph = _canonical(expected, stereo=False)
    center = np.asarray(center, dtype=float)
    size = np.broadcast_to(np.asarray(size, dtype=float), (3,))
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError('Box center/size must contain finite coordinates and positive sizes')

    pose_path = Path(pose_path)
    text = pose_path.read_text(encoding='utf-8')
    try:
        affinities = [float(value) for value in re.findall(r'^REMARK VINA RESULT:\s+(\S+)', text, re.M)]
    except ValueError as exc:
        raise ValueError('Invalid Vina affinity') from exc
    if not affinities or not np.isfinite(affinities).all():
        raise ValueError('Missing/nonfinite Vina affinity')
    sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(text, skip_typing=True))
    if failures or not sdf:
        raise ValueError(f'Pose export failed: {failures}')
    path = pose_path.with_suffix('.sdf')
    temporary = path.with_suffix('.sdf.tmp')
    temporary.write_text(sdf, encoding='utf-8')
    replace_file(temporary, path)
    poses = list(Chem.SDMolSupplier(str(path), removeHs=True))
    if len(poses) != len(affinities) or any(m is None for m in poses):
        raise ValueError('Pose/score count mismatch or invalid exported structure')

    protein = _protein_data(*_cache_key(receptor_pdb))
    lipid = _lipid_data(*_cache_key(lipid_sdf))
    modes = []
    for index, (mol, affinity) in enumerate(zip(poses, affinities), start=1):
        if not np.isfinite(mol.GetConformer().GetPositions()).all():
            raise ValueError(f'Nonfinite coordinates in mode {index}')
        if _canonical(mol, stereo=False) != expected_graph:
            raise ValueError(f'Chemical graph mismatch in mode {index}')
        if _canonical(mol) != expected_smiles:
            raise ValueError(f'Encoded stereochemistry mismatch in mode {index}')
        check = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(check, confId=0, replaceExistingTags=True)
        if _canonical(check) != expected_smiles:
            raise ValueError(f'3D stereochemistry mismatch in mode {index}')
        issues = geometry_issues(mol, 0, expected_smiles)
        metrics = _environment_metrics(mol, protein, lipid, center, size)
        flags = []
        if metrics['heavy_pairs_below_2A']:
            flags.append('protein_heavy_clash_below_2A')
        if metrics['lipid_heavy_pairs_below_2A']:
            flags.append('lipid_heavy_clash_below_2A')
        if metrics['outside_requested_box']:
            flags.append('outside_requested_box')
        elif metrics['within_1A_of_requested_box_edge']:
            flags.append('box_boundary_within_1A')
        if issues:
            flags.append('intramolecular_geometry_issue')
        modes.append({'mode': index, 'affinity_kcal_mol': affinity, **metrics,
                      'geometry_issues': issues, 'quality_flags': flags,
                      'quality_status': 'review' if flags else 'pass',
                      'severe_quality_flags': [f for f in flags if f != 'box_boundary_within_1A']})
    rmsds = [fixed_frame_rmsd(mol, poses[0]) for mol in poses]
    top1 = modes[0]
    flags = list(top1['quality_flags'])
    if any(b < a - 0.001 for a, b in zip(affinities, affinities[1:])):
        flags.append('vina_mode_affinity_order_inconsistent')
    return {**top1,
            'quality_status': 'review' if flags else 'pass', 'quality_flags': flags,
            'all_pose_affinity_kcal_mol': affinities, 'num_modes': len(poses),
            'all_pose_rmsd_to_top1_A': rmsds, 'modes': modes,
            'input_state_smiles': expected_smiles, 'input_unspecified_stereo': [],
            'chemical_identity_verified': True, 'stereochemistry_3d_verified': True,
            'stereochemistry_policy': 'Specified stable input stereo required; 3D reassignment must preserve it.',
            'pose_sdf': str(path.resolve()),
            'quality_scope': 'Top Vina mode; no parent-pose RMSD or contact-recall requirement.',
            'box_boundary_definition': 'Heavy atom margin to requested box faces; negative is outside.',
            'rmsd_method': 'Symmetry-corrected heavy atom RMSD in receptor frame, no alignment; within molecule only.'}
