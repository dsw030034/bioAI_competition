"""Chemical and provenance checks for the experimental LPE diagnostic model."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem

SCRIPT = Path(__file__).with_name('diagnose_nav18_lipid.py')
spec = importlib.util.spec_from_file_location('lipid_diagnostic', SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class LipidChemicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cif = diagnostic.DEFAULT_OUTPUT / 'LPE_ccd.cif'
        if not cls.cif.exists():
            raise RuntimeError('Run diagnose_nav18_lipid.py --prepare-only first')
        cls.models = {state: diagnostic.build_lipid(cls.cif, state) for state in diagnostic.STATES}

    def test_head_states_and_carbon_cap(self):
        for state, (mol, _, _) in self.models.items():
            atoms = {a.GetProp('ccd_atom_id'): a for a in mol.GetAtoms() if a.HasProp('ccd_atom_id')}
            self.assertEqual(atoms['N'].GetFormalCharge(), 1)
            self.assertEqual(atoms['N'].GetDegree(), 4)
            self.assertTrue(all(a.GetAtomicNum() == 6 for a in atoms['N'].GetNeighbors()))
            self.assertEqual(atoms['C16'].GetDegree(), 4)
            self.assertEqual(sum(a.GetAtomicNum() == 1 for a in atoms['C16'].GetNeighbors()), 3)
            self.assertEqual(sum(a.GetAtomicNum() == 1 for a in atoms['O32'].GetNeighbors()), int(state == 'ccd_acid_plus1'))
            self.assertEqual(Chem.GetFormalCharge(mol), int(state == 'ccd_acid_plus1'))

    def test_heavy_coordinates_and_rigid_receptor(self):
        deposited = diagnostic.observed_lipid()
        original = diagnostic.RECEPTOR.read_text().splitlines()
        for state, (mol, _, _) in self.models.items():
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 1:
                    np.testing.assert_array_equal(mol.GetConformer().GetAtomPosition(atom.GetIdx()),
                                                  deposited[atom.GetProp('ccd_atom_id')])
            lines = (diagnostic.DEFAULT_OUTPUT / state / 'receptor_with_lipid.pdbqt').read_text().splitlines()
            self.assertEqual(lines[:len(original)], original)
            self.assertTrue(all(line.startswith('ATOM  ') for line in lines[len(original):]))
            serials = [int(line[6:11]) for line in lines if line.startswith(('ATOM  ', 'HETATM'))]
            self.assertEqual(len(serials), len(set(serials)))

    def test_unexpected_missing_heavy_atom_refused(self):
        observed = diagnostic.observed_lipid()
        del observed['C14']
        with patch.object(diagnostic, 'observed_lipid', return_value=observed):
            with self.assertRaises(AssertionError):
                diagnostic.build_lipid(self.cif, 'zwitterion')


if __name__ == '__main__':
    unittest.main()
