"""Checked ligand conformers and residue-template receptor preparation."""
from __future__ import annotations

import hashlib
from pathlib import Path

from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem

from .ligand_quality import prepare_candidate
from .runtime import read_json, sha256, versions, write_json

AD4_TO_ELEMENT = {
    'A': 'C', 'C': 'C', 'N': 'N', 'NA': 'N', 'O': 'O', 'OA': 'O',
    'S': 'S', 'SA': 'S', 'Cl': 'Cl', 'F': 'F', 'Br': 'Br', 'I': 'I', 'P': 'P',
    'HD': 'H', 'HS': 'H',
}


class PreparationReviewRequired(ValueError):
    pass


def prep_ligand(smiles, out_pdbqt=None, seed=42, *, candidate_id=None, num_confs=20,
                resume=False, neutral_qdn_reference=False):
    """Select the lowest-energy valid conformer from a checked input-state pool.

    Energy is compared only within this state. Scoring remains the top Vina pose
    from one selected input conformer per target, using the original formula.
    """
    if num_confs < 1:
        raise ValueError('num_confs must be positive')
    candidate_id = candidate_id or hashlib.sha256(smiles.encode()).hexdigest()[:16]
    path = Path(out_pdbqt or f'{candidate_id}.pdbqt')
    record_path = path.with_suffix('.preparation.json')
    signature = {'smiles': smiles, 'candidate_id': candidate_id, 'seed': seed,
                 'num_confs': num_confs, 'neutral_qdn_reference': neutral_qdn_reference,
                 'versions': versions(), 'preparer_sha256': sha256(__file__),
                 'quality_sha256': sha256(Path(__file__).with_name('ligand_quality.py'))}
    if record_path.exists():
        previous = read_json(record_path)
        if not resume or previous['signature'] != signature:
            raise ValueError('Ligand already exists or preparation inputs changed')
        if previous['status'] == 'ok':
            for name, digest in previous['files'].items():
                if sha256(path.parent / name) != digest:
                    raise ValueError(f'Prepared ligand artifact changed: {name}')
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {'signature': signature, 'status': 'running'}
    write_json(record_path, result)
    try:
        candidate = {'candidate_id': candidate_id, 'smiles': smiles}
        if neutral_qdn_reference:
            candidate['neutral_qdn_reference'] = True
        state, records, mol, kept = prepare_candidate(candidate, num_confs=num_confs,
                                                     max_keep=1, random_seed=seed)
        result.update(state=state, conformers=records)
        if not kept:
            raise PreparationReviewRequired(state['status'])
        selected = Chem.Mol(mol)
        conformer = Chem.Conformer(mol.GetConformer(kept[0]['rdkit_conf_id']))
        selected.RemoveAllConformers()
        selected.AddConformer(conformer, assignId=True)
        for key in ('candidate_id', 'state_id', 'state_smiles'):
            selected.SetProp(key, str(state[key]))
        selected.SetProp('conformer_id', kept[0]['conformer_id'])
        sdf = path.with_suffix('.sdf')
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(selected)
        setups = MoleculePreparation().prepare(selected)
        if len(setups) != 1:
            raise ValueError('Unexpected multiple Meeko ligand setups')
        text, ok, error = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise ValueError(error)
        path.write_text(text, encoding='utf-8')
        result.update(status='ok', selected_conformer_id=kept[0]['conformer_id'],
                      files={p.name: sha256(p) for p in (path, sdf)})
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(record_path, result)
    return path


def prep_receptor(protein_pdb, out_pdbqt=None, center=None, radius=None, *, chain=None,
                  size=24.0, resume=False):
    """Compatibility entry with explicit chain/box and no per-atom sphere cropping."""
    from .receptors import prepare_receptor
    if chain is None or center is None:
        raise ValueError('Receptor preparation requires explicit chain and center')
    if radius is not None:
        raise ValueError('Per-atom radius cropping is unsupported; use full selected chain')
    destination = Path(out_pdbqt or Path(protein_pdb).with_suffix('.pdbqt'))
    prepared = prepare_receptor(protein_pdb, destination.parent / (destination.stem + '_preparation'),
                                chain, center, size, resume)
    destination.write_bytes(prepared.read_bytes())
    return destination
