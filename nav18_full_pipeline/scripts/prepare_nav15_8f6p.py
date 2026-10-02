"""Prepare rat 8F6P/ranolazine with the same audited CCD/experimental-pose checks."""
import argparse
from prepare_nav15_6uz0 import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/nav15_multireference_20261002/rat_8f6p')
    args = parser.parse_args()
    run(args.output, pdb_id='8F6P', ligand_id='XHO')
