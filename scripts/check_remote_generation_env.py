"""Capture reproducible evidence for a configured REINVENT GPU environment."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from rdkit import Chem
    from reinvent.Reinvent import main_script

    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {p: metadata.version(p) for p in ("reinvent", "torch", "torchvision", "rdkit", "numpy", "pandas")},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
              "reinvent_entry_imported": callable(main_script), "rdkit_parse": Chem.MolFromSmiles("CCO") is not None}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    x = torch.ones((128, 128), device="cuda")
    report["cuda_matmul_ok"] = bool(torch.all(x @ x == 128).item())
    raw = args.prior.read_bytes()
    report["prior_sha256"] = hashlib.sha256(raw).hexdigest()
    report["prior_md5"] = hashlib.md5(raw).hexdigest()
    report["official_prior_checksum_match"] = report["prior_md5"] == "78b75e478bfd7dcccee8ece010bc0b07"
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report["pip_check"] = {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if check.returncode or not report["cuda_matmul_ok"] or not report["official_prior_checksum_match"]:
        raise RuntimeError("Environment checks failed")


if __name__ == "__main__":
    main()
