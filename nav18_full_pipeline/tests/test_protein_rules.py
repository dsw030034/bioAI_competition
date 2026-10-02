from navgen.protein_rules import ProteinWordScorer, extract_protein_words


def test_protein_word_scorer_returns_interpretable_matches():
    sequence = "MKWVTFISLLFLFSSAYSRASTGKKIGYSARDKQLEAAGVDAL"
    words = extract_protein_words(sequence, word_size=5)
    result = ProteinWordScorer(sequence).score("COc1ccccc1C(=O)Nc1ccccc1")

    assert words
    assert result["protein_word_score"] >= 0
    assert isinstance(result["matched_rules"], list)
