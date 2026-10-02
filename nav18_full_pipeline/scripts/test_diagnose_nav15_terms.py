"""Scientific contract checks for isolated Vina term attribution."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_nav15_terms import Atom, assign_xs, pair_terms, fast_pair_terms, parse_energies, TERMS, WEIGHTS, read_json


def atom(serial, ad, x, xs=''):
    return Atom(serial, ad, 'B:TEST:1', np.array([x,0.,0.]), ad, xs)


class VinaTermChecks(unittest.TestCase):
    def test_amide_nitrogen_donor_not_acceptor(self):
        atoms = [atom(1,'N',0),atom(2,'HD',1),atom(3,'C',-1.4)]
        assign_xs(atoms)
        self.assertEqual(atoms[0].xs,'N_D')
        self.assertEqual(atoms[2].xs,'C_P')
        oxygen = atom(4,'OA',3,'O_A')
        self.assertGreater(pair_terms(atoms[0],oxygen,3)[4],0)
        self.assertEqual(pair_terms(atoms[0],atom(5,'N',3,'N_D'),3)[4],0)

    def test_hydrophobic_excludes_hetero_bound_carbon(self):
        hydro = atom(1,'C',0,'C_H')
        polar = atom(2,'C',4,'C_P')
        self.assertEqual(pair_terms(hydro,polar,4)[3],0)
        self.assertEqual(pair_terms(hydro,atom(3,'C',4,'C_H'),4)[3],1)

    def test_cutoff_hydrogens_and_repulsion(self):
        a,b = atom(1,'C',0,'C_H'),atom(2,'C',3,'C_H')
        self.assertGreater(pair_terms(a,b,3)[2],0)
        self.assertEqual(pair_terms(a,b,4)[2],0)
        self.assertTrue(np.all(pair_terms(a,b,8)==0))
        self.assertTrue(np.all(pair_terms(a,atom(3,'HD',1,'H_P'),1)==0))

    def test_fast_lookup_is_r_squared_quantized(self):
        a,b = atom(1,'C',0,'C_H'),atom(2,'C',4,'C_H')
        expected = (pair_terms(a,b,4)+pair_terms(a,b,np.sqrt(16+1/32)))*.5*WEIGHTS
        np.testing.assert_allclose(fast_pair_terms(a,b,16.0001),expected,rtol=1e-14)

    def test_original_binary_agreement_and_conservation(self):
        report_path = Path(__file__).resolve().parents[1]/'outputs/nav15_terms_20261002/report.json'
        if not report_path.exists():
            self.skipTest('Run the documented fixed-pose diagnostic for binary integration check')
        report = read_json(report_path)
        self.assertTrue(report['all_verified'])
        self.assertTrue(report['production_code_unchanged'])
        self.assertGreaterEqual(report['attempted_score_only'],8)
        for row in report['summary']:
            with self.subTest(pose=row['pose']):
                self.assertLessEqual(abs(sum(row[t] for t in TERMS)-row['vina_intermolecular_kcal_mol']),.001)
                self.assertAlmostEqual(row['vina_affinity_kcal_mol'],row['vina_intermolecular_kcal_mol']+row['vina_internal_kcal_mol']+row['vina_torsional_kcal_mol']-row['vina_unbound_kcal_mol'],delta=.002)


if __name__ == '__main__':
    unittest.main()
