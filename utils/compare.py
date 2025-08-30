import difflib

def compare_strings(reference, spoken):
    results = []

    correct_chars_total = 0
    total_chars_total = 0

    for i, paragraph in enumerate(reference):

        matcher = difflib.SequenceMatcher(None, paragraph, spoken[i])
        result = []
        correct_chars = 0
        total_chars = len(paragraph)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            ref_part = reference[i1:i2]
            spoken_part = spoken[i][j1:j2]

            if tag == 'equal':
                result.append({"text": spoken_part, "type": "correct"})
                correct_chars += (i2 - i1)
            elif tag in ('replace', 'insert', 'delete'):
                result.append({"text": spoken_part, "type": "incorrect"})

        correct_chars_total += correct_chars
        total_chars_total += total_chars

        if result[-1]["text"] == "":
            result = result[:-1]

        results.append(result)
    
    accuracy = correct_chars_total / total_chars_total if total_chars_total > 0 else 0
    return results, accuracy

def check_missing_words(reference, spoken):
    missing_words = []

    for i, paragraph in enumerate(reference):

        paragraph = paragraph.lower().replace(".", "").replace(",", "").split(" ")
        spoken_paragraph = spoken[i].lower().replace(".", "").replace(",", "").split(" ")

        if len(paragraph) > len(spoken_paragraph):
            missing_words.append(-1)
        elif len(spoken_paragraph) > len(paragraph):
            missing_words.append(1)
        else:
            missing_words.append(0)

    return missing_words
