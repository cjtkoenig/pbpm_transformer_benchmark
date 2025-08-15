import textdistance

def normalized_damerau_levenshtein(predicted_sequence, true_sequence) -> float:
    """Return similarity in [0,1] where 1.0 means identical sequences."""
    edit_distance = textdistance.damerau_levenshtein.distance(predicted_sequence, true_sequence)
    max_length = max(len(predicted_sequence), len(true_sequence), 1)
    return 1.0 - edit_distance / max_length
