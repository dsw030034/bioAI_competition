from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairingRule:
    word: str
    fragment: str
    score: float
    rationale: str


def load_protein_sequence(path: str | Path) -> str:
    sequence_path = Path(path)
    if not sequence_path.exists():
        raise FileNotFoundError(f"Protein sequence file not found: {sequence_path}")
    sequence = "".join(
        line.strip()
        for line in sequence_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    if not sequence or re.search(r"[^ACDEFGHIKLMNPQRSTVWY]", sequence):
        raise ValueError(f"Protein sequence contains invalid amino-acid symbols: {sequence_path}")
    return sequence


def extract_protein_words(sequence: str, word_size: int = 5) -> list[str]:
    if word_size < 3 or word_size > len(sequence):
        raise ValueError("word_size must be between 3 and the sequence length")
    return sorted({sequence[i : i + word_size] for i in range(len(sequence) - word_size + 1)})


def _word_class(word: str) -> str:
    hydrophobic = sum(residue in "AILMFWVY" for residue in word) / len(word)
    charged = sum(residue in "DEKR" for residue in word) / len(word)
    polar = sum(residue in "STNQH" for residue in word) / len(word)
    aromatic = sum(residue in "FWY" for residue in word) / len(word)
    if aromatic >= 0.2:
        return "aromatic"
    if charged >= 0.4:
        return "charged"
    if polar >= 0.4:
        return "polar"
    if hydrophobic >= 0.6:
        return "hydrophobic"
    return "mixed"


def build_rules(sequence: str, word_size: int = 5) -> list[PairingRule]:
    rules: list[PairingRule] = []
    for word in extract_protein_words(sequence, word_size):
        word_class = _word_class(word)
        if word_class in {"aromatic", "hydrophobic"}:
            rules.extend(
                [
                    PairingRule(word, "aromatic", 0.75, "hydrophobic/aromatic complementarity"),
                    PairingRule(word, "halogen", 0.45, "non-polar contact and halogen-compatible environment"),
                ]
            )
        elif word_class == "charged":
            rules.extend(
                [
                    PairingRule(word, "basic", 0.8, "opposite-charge or cation-pi complementarity"),
                    PairingRule(word, "polar", 0.55, "polar hydrogen-bond complementarity"),
                ]
            )
        elif word_class == "polar":
            rules.extend(
                [
                    PairingRule(word, "polar", 0.8, "hydrogen-bond donor/acceptor complementarity"),
                    PairingRule(word, "carbonyl", 0.6, "carbonyl-compatible polar contact"),
                ]
            )
    return rules


def _fragment_classes(smiles: str) -> set[str]:
    classes: set[str] = set()
    if re.search(r"c1|c2|C1=CC|C2=CC", smiles):
        classes.add("aromatic")
    if re.search(r"F|Cl|Br|I", smiles):
        classes.add("halogen")
    if re.search(r"[Nn]", smiles):
        classes.add("basic")
    if re.search(r"O|S", smiles):
        classes.add("polar")
    if re.search(r"C\(=O\)|C=O", smiles):
        classes.add("carbonyl")
    return classes


class ProteinWordScorer:
    """Interpretable prototype inspired by PWRules, not a trained PWRules model."""

    def __init__(self, sequence: str, word_size: int = 5):
        self.sequence = sequence
        self.word_size = word_size
        self.rules = build_rules(sequence, word_size)

    def score(self, smiles: str) -> dict[str, Any]:
        fragment_classes = _fragment_classes(smiles)
        matches = [
            rule
            for rule in self.rules
            if rule.fragment in fragment_classes
        ]
        if not matches:
            return {
                "protein_word_score": 0.0,
                "matched_rule_count": 0,
                "matched_rules": [],
                "protein_word_coverage": 0.0,
            }
        best_by_word: dict[str, PairingRule] = {}
        for rule in matches:
            current = best_by_word.get(rule.word)
            if current is None or rule.score > current.score:
                best_by_word[rule.word] = rule
        selected = sorted(best_by_word.values(), key=lambda rule: rule.score, reverse=True)
        mean_rule_score = sum(rule.score for rule in selected) / len(selected)
        class_coverage = min(1.0, len(fragment_classes) / 5.0)
        score = 0.6 * (mean_rule_score / 0.8) + 0.4 * class_coverage
        return {
            "protein_word_score": round(min(1.0, score), 6),
            "matched_rule_count": len(selected),
            "matched_rules": [
                {
                    "word": rule.word,
                    "fragment": rule.fragment,
                    "score": rule.score,
                    "rationale": rule.rationale,
                }
                for rule in selected[:10]
            ],
            "protein_word_coverage": round(len(selected) / max(len(self.rules), 1), 6),
        }

    def rules_as_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "word": rule.word,
                "fragment": rule.fragment,
                "score": rule.score,
                "rationale": rule.rationale,
            }
            for rule in self.rules
        ]
