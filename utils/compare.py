import difflib

def compare_strings(reference: str, spoken: str):
    matcher = difflib.SequenceMatcher(None, reference, spoken)
    result = []
    correct_chars = 0
    total_chars = len(reference)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_part = reference[i1:i2]
        spoken_part = spoken[j1:j2]

        if tag == 'equal':
            result.append({"text": spoken_part, "type": "correct"})
            correct_chars += (i2 - i1)
        elif tag in ('replace', 'insert', 'delete'):
            result.append({"text": spoken_part, "type": "incorrect"})

    accuracy = correct_chars / total_chars if total_chars > 0 else 0

    return result, accuracy

def check_missing_words(reference: str, spoken):
    reference = reference.lower().replace(".", "").replace(",", "").split(" ")
    spoken = spoken.lower().replace(".", "").replace(",", "").split(" ")

    if len(reference) > len(spoken):
        return -1
    elif len(spoken) > len(reference):
        return 1
    else: return 0

    # matcher = difflib.SequenceMatcher(None, reference, spoken)
    # missing = []

    # for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    #     if tag == 'delete':
    #         missing.extend((reference[i1:i2], i1))
    
    return missing
