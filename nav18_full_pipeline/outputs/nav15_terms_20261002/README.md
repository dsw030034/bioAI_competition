# Nav1.5 original Vina term diagnostic (2026-10-02)

This is fixed-coordinate attribution of existing poses. Production scoring, weights, receptor and candidates were not changed. No new global docking was performed. Sixteen unique poses were checked: eight previous fixed/local inputs and eight matched-initial-geometry local endpoints. All reconstructed intermolecular sums agree with the unchanged Vina 1.2.7 binary `--score_only` output. Maximum error: 0.000342 kcal/mol, within three-decimal log rounding.

## What is decomposed

- Five original terms: gauss1, gauss2, repulsion, hydrophobic, non-directional hydrogen bond.
- Their sum is **intermolecular energy**, not affinity. Original Vina affinity/internal/torsional/unbound entries remain separate in `summary.csv`.
- No weight changes. The independent Python implementation reproduces XS typing, squared-distance fast lookup, 8 A cutoff and per-ligand-atom positive-energy curl. Pair/residue attribution is emitted only after agreement with the unchanged binary.
- Upstream source and Apache 2.0 notices: `upstream_vina_1.2.7/`; URLs and SHA256: `manifest.json`.
- Reproduce from pipeline root: `python -B scripts/diagnose_nav15_terms.py --output outputs/NEW_TERMS_DIRECTORY`.
- Tests: `python -B -m unittest discover -s scripts -p test_diagnose_nav15_terms.py -v`; five passed, including integration against all 16 original-binary outputs.

## Matched initial internal geometry

Values below are **wrong site minus native site** in intermolecular kcal/mol. Negative favors the wrong region. Two sites use the same starting internal geometry/hydrogen scheme. Local optimization can still change torsions, so final internal geometries are not constrained identical. Native/wrong labels refer to starting geometry source.

| Charge/initial geometry | gauss1 | gauss2 | Repulsion | Hydrophobic | H-bond | Sum |
|---|---:|---:|---:|---:|---:|---:|
| Neutral/native | -0.693 | -0.361 | +2.186 | -0.226 | -0.369 | **+0.536** |
| Neutral/wrong | -1.248 | -0.628 | +0.769 | -1.709 | +0.019 | **-2.797** |
| +1/native | -0.509 | -0.515 | +1.852 | -0.144 | -0.306 | **+0.377** |
| +1/wrong | -1.487 | -0.597 | +0.204 | -0.946 | -0.243 | **-3.068** |

With native starting geometry, additional repulsion in the other region offsets its attractive contact gains. With docking-derived geometry, Gaussian and hydrophobic gains favor the other region. The original unmatched-geometry comparison therefore cannot establish a preparation-independent preference for the other site. This is not successful global redocking.

Gaussian terms should not all be called hydrophobic energy. These are model attributions, not physical experimental energy decomposition. The original Vina function contains no separate explicit Coulomb term and these results cannot establish the real effect of formal charge.

## Residue attribution

The other region gains contacts involving PHE1459, PHE934, VAL930, and PHE1418/LEU931/LEU895. The native region is favored by PHE1760, PHE402, VAL1763 and nearby contacts. Residue identifiers retain original chain B numbering. For the neutral/wrong starting geometry, wrong-minus-native residue contributions include PHE1459 -2.000, PHE934 -1.578, PHE1760 +1.380 and PHE402 +0.825 kcal/mol. Full pair/residue tables are retained. These do not establish biological binding hotspots or independently select residues for flexible docking.

## Grid-boundary anomaly: distinct from ordinary rounding

Three +1 local saved poses have a heavy-atom x extent of 12.000125 A relative to the box center, exceeding the original 12 A half-width by 0.000125 A. Fixed scoring under the original box refuses these saved coordinates. For these rows only, the fixed-score box was padded to 24.75 A without moving any atom or changing the scorer. Each row records its score-box size.

For +1/native geometry -> wrong site, the original local log reports affinity +0.763/inter +0.964. Rescoring the identical saved coordinates with padding gives affinity -6.376/inter -8.053. This is far beyond ordinary printed-coordinate drift. The +1/wrong -> wrong endpoint changes only from about -8.273 to -8.270.

The upstream code supports a boundary-penalty explanation: `vina.cpp` computes maps with slope=1e6; `non_cache.cpp` adds out-of-bound distance times slope to intermolecular energy; `cache.h` allows `is_in_grid` a default margin of 0.0001 A. A tiny overshoot can pass the grid check yet incur several kcal/mol. Exact unrounded coordinates were not preserved, so this diagnostic cannot numerically recover the entire original discrepancy. Wider-box local optimization is being evaluated separately. The +0.763 endpoint must not be used as evidence that this molecular geometry cannot bind.

Additional cache.h provenance fetched after the run manifest is in `boundary_source_note.json`. The completed script hash still matches its manifest. No candidate gate was released and no candidate was removed.
