"""Explicit, UTF-8 file contracts shared by the pipeline and external scorers."""

import csv
import hashlib
import json
from pathlib import Path


def read_json(path):
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_smiles(path):
    """CSV column names are case insensitive (including REINVENT's SMILES)."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
            columns = {c.lower(): c for c in reader.fieldnames or []}
            if "smiles" not in columns:
                raise ValueError(f"{path}: CSV requires a smiles column")
            for row in reader:
                yield row[columns["smiles"]].strip()
        else:
            for line in handle:
                if line.strip() and not line.lstrip().startswith("#"):
                    yield line.split()[0]


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

