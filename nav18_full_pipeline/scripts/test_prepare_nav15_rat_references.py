"""Artifact-backed checks for the explicitly audited rat reference preparations."""
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
from rdkit import Chem
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_nav15_6uz0 import ccd_reference
from scgd.geometry import pdb_atoms
from scgd.runtime import read_json, sha256


BASE=Path(__file__).resolve().parents[1]/'outputs/nav15_multireference_20261002'


class RatReferencePreparationTests(unittest.TestCase):
    def test_ccd_graph_and_observed_coordinates_are_preserved(self):
        for panel,entry,ccd in [('rat_6uz0','6UZ0','K4D'),('rat_8f6p','8F6P','XHO')]:
            with self.subTest(panel=panel):
                folder=BASE/panel
                rebuilt,audit=ccd_reference(folder/'sources'/f'{ccd}.cif',folder/'sources'/f'{entry}.pdb',ccd)
                saved=Chem.SDMolSupplier(str(folder/'reference.sdf'))[0]
                self.assertEqual(Chem.MolToSmiles(rebuilt),Chem.MolToSmiles(saved))
                np.testing.assert_array_equal(rebuilt.GetConformer().GetPositions(),saved.GetConformer().GetPositions())
                self.assertTrue(all(row['ccd']==row['experimental_3d'] for row in audit['stereochemistry']))

    def test_incomplete_experimental_ligand_is_rejected(self):
        folder=BASE/'rat_6uz0'
        lines=(folder/'sources/6UZ0.pdb').read_text().splitlines()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'incomplete.pdb'
            path.write_text('\n'.join(line for line in lines if not
                                     (line.startswith('HETATM') and line[17:20]=='K4D' and line[12:16].strip()=='C14')))
            with self.assertRaisesRegex(ValueError,'Incomplete ligand'):
                ccd_reference(folder/'sources/K4D.cif',path)

    def test_new_oxt_repair_did_not_move_original_atoms(self):
        folder=BASE/'rat_6uz0'
        old=pdb_atoms(folder/'receptor/selected.pdb')
        new=pdb_atoms(folder/'receptor_oxt_repaired/prepared.pdb')
        self.assertTrue(set(old).issubset(new))
        for key,coords in old.items():
            np.testing.assert_array_equal(coords,new[key])
        audit=read_json(folder/'receptor_oxt_repaired/audit.json')
        self.assertEqual(6,len(audit['oxt_repairs']))
        for row in audit['oxt_repairs']:
            key=tuple(row['key'])
            self.assertNotIn(key,old)
            self.assertEqual('OXT',key[-1])
            self.assertGreater(row['new_O_OXT_A'],1.8)
            self.assertGreater(row['new_OXT_minimum_other_heavy_excluding_bonded_C_A'],1.8)

    def test_reviewed_links_do_not_delete_peptides_or_disulfides(self):
        folder=BASE/'rat_6uz0'
        removed=read_json(folder/'receptor_oxt_repaired/reviewed_false_bonds.json')
        self.assertEqual(10,len(removed))
        for row in removed:
            self.assertNotEqual({'C','N'},set(row['atom_names']))
            self.assertNotEqual(['SG','SG'],row['atom_names'])
            self.assertGreater(min(row['atom_minimum_ligand_distances_A']),8)
        manifest=read_json(folder/'prepare_manifest.json')
        self.assertEqual(manifest['script_sha256'],sha256(folder/'provenance/prepare_nav15_6uz0.py'))

    def test_8f6p_exclusions_are_remote_and_kept_observed_atoms_fixed(self):
        folder=BASE/'rat_8f6p'
        old=pdb_atoms(folder/'receptor/selected.pdb')
        final=pdb_atoms(folder/'receptor_distal_filtered/prepared.pdb')
        audit=read_json(folder/'receptor_distal_filtered/audit.json')
        exclusions=audit['explicit_distant_template_exclusions']
        self.assertEqual(15,len(exclusions))
        for row in exclusions.values():
            self.assertGreater(row['minimum_distance_to_box_A'],8)
        for key,coords in old.items():
            if ':'.join(key[:2]) not in exclusions:
                np.testing.assert_array_equal(coords,final[key])
        self.assertEqual([],audit['new_heavy_atoms_within_8A_of_reference'])

    def test_8f6p_source_disulfides_are_preserved(self):
        audit=read_json(BASE/'rat_8f6p/disulfide_preservation_audit.json')
        self.assertEqual(4,len(audit))
        for row in audit:
            self.assertTrue(row['both_observed_atoms_preserved'])
            self.assertEqual(row['source_distance_A'],row['prepared_distance_A'])


if __name__=='__main__': unittest.main()
