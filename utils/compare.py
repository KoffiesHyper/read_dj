import difflib
import re
from difflib import SequenceMatcher

def normalize_text(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    return text.split()

def compare_strings(reference, spoken):
    results = []

    correct_words_total = 0
    words_total = 0

    for i, paragraph in enumerate(reference):

        if spoken[i] == "empty":
            results.append([])
            continue

        ref_words = normalize_text(paragraph)
        hyp_words = normalize_text(spoken[i])

        matcher = SequenceMatcher(None, ref_words, hyp_words)
        alignment = []

        correct_words = 0
        words = len(ref_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for r, h in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                    alignment.append((r, h, "correct"))
                    correct_words += 1
            elif tag == "replace":
                for r, h in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                    alignment.append((r, h, "substitution"))
            elif tag == "delete":
                for r in ref_words[i1:i2]:
                    alignment.append((r, None, "deletion"))
            elif tag == "insert":
                for h in hyp_words[j1:j2]:
                    alignment.append((None, h, "insertion"))
            
            correct_words_total += correct_words
            words_total += words
        
        results.append(alignment)

        # matcher = difflib.SequenceMatcher(None, paragraph, spoken[i])
        # result = []
        # correct_chars = 0
        # total_chars = len(paragraph)

        # for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        #     ref_part = reference[i1:i2]
        #     spoken_part = spoken[i][j1:j2]

        #     if tag == 'equal':
        #         result.append({"text": spoken_part, "type": "correct"})
        #         correct_chars += (i2 - i1)
        #     elif tag in ('replace', 'insert', 'delete'):
        #         result.append({"text": spoken_part, "type": "incorrect"})

        # correct_chars_total += correct_chars
        # total_chars_total += total_chars

        # if result[-1]["text"] == "":
        #     result = result[:-1]

        # results.append(result)
    
    accuracy = correct_words_total / words_total if correct_words_total > 0 else 0
    return results, accuracy

def check_missing_words(reference, spoken):
    missing_words = []

    for i, paragraph in enumerate(reference):

        if paragraph == "empty":
            missing_words.append(0)

        paragraph = paragraph.lower().replace(".", "").replace(",", "").split(" ")
        spoken_paragraph = spoken[i].lower().replace(".", "").replace(",", "").split(" ")

        if len(paragraph) > len(spoken_paragraph):
            missing_words.append(-1)
        elif len(spoken_paragraph) > len(paragraph):
            missing_words.append(1)
        else:
            missing_words.append(0)

    return missing_words
