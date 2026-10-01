"""Official modern ShEPhERD conditional API; run in its separate GPU environment.

Reference: coleygroup/shepherd examples/conditional_generation.ipynb.
No dependency on benchmark's legacy imports. No weights downloaded in preflight.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--reference-source", required=True, help="Experimental/docking pose provenance; never inferred")
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=100, help="Fixed RAW sampling budget")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    if min(a.samples, a.batch_size, a.steps) <= 0:
        p.error("samples, batch-size and steps must be positive")
    from navgen.config import load_config
    from navgen.generation import load_seed_smiles
    from navgen.experiment import read_reference
    cfg = load_config(a.config)
    ref = read_reference(a.reference, load_seed_smiles(cfg["seed_file"])[0])
    required = ["torch", "shepherd", "shepherd_score"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not shutil.which("xtb"):
        missing.append("xtb executable")
    report = {"reference_identity_checked": True, "reference_source": a.reference_source,
              "missing_dependencies": missing, "samples": a.samples, "steps": a.steps,
              "reference_sha256": hashlib.sha256(Path(a.reference).read_bytes()).hexdigest()}
    if a.preflight:
        print(json.dumps(report, indent=2))
        return 2 if missing else 0
    if missing:
        raise RuntimeError("Use an isolated ShEPhERD environment; missing: " + ", ".join(missing))
    import numpy as np
    import torch
    from rdkit import Chem
    from shepherd import load_model
    from shepherd.inference import generate
    from shepherd_score.container import Molecule
    from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb, update_mol_coordinates
    from shepherd_score.evaluations.evaluate.pipelines import ConditionalEvalPipeline
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report.update(status="running", arguments=vars(a), versions={m: importlib.metadata.version(m.replace('_','-')) for m in required})
    def manifest():
        (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest()
    try:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        ref = Chem.AddHs(ref, addCoords=True)
        # Single-point charges keep the reference heavy-atom pose; no relaxation.
        charges = charges_from_single_point_conformer_with_xtb(ref, solvent="water", num_cores=1,
                    charge=Chem.GetFormalCharge(ref), temp_dir=str(out))
        center = ref.GetConformer().GetPositions().mean(axis=0)
        ref = update_mol_coordinates(ref, ref.GetConformer().GetPositions() - center)
        with Chem.SDWriter(str(out / "reference_centered.sdf")) as writer:
            writer.write(ref)
        model = load_model("mosesaq", device=a.device)
        profile = Molecule(ref, num_surf_points=model.params['dataset']['x3']['num_points'],
                          probe_radius=model.params['dataset']['probe_radius'],
                          partial_charges=charges, pharm_multi_vector=False)
        samples = []
        for start in range(0, a.samples, a.batch_size):
            batch = generate(model_pl=model, batch_size=min(a.batch_size, a.samples-start),
                N_x1=max(5, ref.GetNumAtoms() + (-4, 0, 4)[(start // a.batch_size) % 3]),
                N_x4=len(profile.pharm_types), unconditional=False,
                inpaint_x3_pos=True, inpaint_x3_x=True, inpaint_x4_pos=True,
                inpaint_x4_direction=True, inpaint_x4_type=True,
                center_of_mass=np.zeros(3), surface=profile.surf_pos, electrostatics=profile.surf_esp,
                pharm_types=profile.pharm_types, pharm_pos=profile.pharm_ancs,
                pharm_direction=profile.pharm_vecs, verbose=True, num_steps=a.steps)
            samples.extend(batch)
            # Preserve every attempt before chemical reconstruction/relaxation.
            import pickle
            with (out / "raw_samples.pkl").open("wb") as handle:
                pickle.dump(samples, handle)
        evaluation_ref = Molecule(ref, num_surf_points=400, probe_radius=1.2,
                                 partial_charges=charges, pharm_multi_vector=False)
        evaluator = ConditionalEvalPipeline(evaluation_ref,
            [(s['x1']['atoms'], s['x1']['positions']) for s in samples], condition="all",
            num_surf_points=400, pharm_multi_vector=False, solvent="water")
        evaluator.evaluate(num_workers=1, verbose=True)
        summary, details = evaluator.to_pandas()
        summary.to_csv(out / "shepherd_summary.csv")
        details.to_csv(out / "shepherd_details.csv")
        count = 0
        with Chem.SDWriter(str(out / "generated_relaxed.sdf")) as writer:
            for i, block in enumerate(evaluator.molblocks_post_opt):
                mol = Chem.MolFromMolBlock(block, removeHs=False) if isinstance(block, str) and block else None
                if mol is not None:
                    mol.SetIntProp("raw_sample_index", i)
                    mol.SetProp("coordinate_frame", "centered_reference; requires receptor alignment/redocking")
                    writer.write(mol)
                    count += 1
        report.update(status="complete", raw_samples=len(samples), relaxed_valid=count,
                      reference_translation=center.tolist(),
                      scope="ligand-profile generation, not pocket-conditioned; no activity claims")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
