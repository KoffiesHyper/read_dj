import re
import subprocess
from .LoadModel import LoadModel
from .Transcribe import Transcribe

from decouple import config

ROOT_PATH = config("ROOT_PATH")
MMS_PATH = config("MMS_PATH")
device = config("device")

class MispronunciationDetection:
    PHONEME_MAP = {
        # Vowels
        'a': 'a', 'ə': 'a', 'ʌ': 'uh', 'æ': 'a', 'ɑ': 'ah', 'e': 'e', 'ɛ': 'e',
        'ɪ': 'i', 'i': 'ee', 'ɒ': 'o', 'ɔ': 'aw', 'ʊ': 'oo', 'u': 'oo',
        'ɜ': 'er',
        'aɪ': 'i', 'aʊ': 'ow', 'eɪ': 'ay', 'oʊ': 'oh', 'ɔɪ': 'oy',
        'ɪə': 'eer', 'eə': 'air', 'ʊə': 'oor',
        # Consonants
        'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'h': 'h', 'j': 'y',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r',
        's': 's', 't': 't', 'v': 'v', 'w': 'w', 'z': 'z',
        'θ': 'th', 'ð': 'th', 'ʃ': 'sh', 'ʒ': 'zh', 'ŋ': 'ng',
        'tʃ': 'ch', 'dʒ': 'j', 'ʍ': 'wh',
        'ʔ': '', 'ɾ': 'tt',
        'd͡ʒ': 'j', 't͡ʃ': 'ch', 't͡s': 'ts', 'd͡z': 'dz',
        # Stress/markers (ignored)
        'ˈ': '', 'ˌ': '', '.': '', '!': '',
        # SA approximations
        'x': 'gh', 'r̩': 'r', 'l̩': 'l', 'm̩': 'm', 'n̩': 'n',
    }

    def __init__(self, model, processor, device=None):
        # Load model & processor
        model = model
        processor = processor

        # Create transcriber
        self.transcriber = Transcribe(model, processor, device=device)

    
    def phonemes_to_letters(self, phoneme_token: str) -> str:
        """Map a single phoneme token to its alphabetic representation."""
        return self.PHONEME_MAP.get(phoneme_token, '')

    
    def _align_phonemes(self, gt_phonemes, pred_phonemes):
        """
        Aligns two sequences of phonemes to find substitutions, insertions, and deletions.
        Returns a list of tuples: (type, gt_phoneme, pred_phoneme)
        type: 'match', 'substitution', 'insertion', 'deletion'
        """
        n = len(gt_phonemes)
        m = len(pred_phonemes)

        # Initialize DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if gt_phonemes[i - 1] == pred_phonemes[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,       # Deletion
                               dp[i][j - 1] + 1,       # Insertion
                               dp[i - 1][j - 1] + cost) # Substitution/Match

        # Backtrack to find alignment
        alignment = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and gt_phonemes[i - 1] == pred_phonemes[j - 1]:
                alignment.append(('match', gt_phonemes[i - 1], pred_phonemes[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append(('substitution', gt_phonemes[i - 1], pred_phonemes[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                alignment.append(('deletion', gt_phonemes[i - 1], None))
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                alignment.append(('insertion', None, pred_phonemes[j - 1]))
                j -= 1
        return alignment[::-1] # Reverse to get correct order

    def find_mispronunciations(self, ground_truth, predicted, original_text):
        original_text_list = original_text.split()
        mispronunciation_espeak_dict = {}
        mispronunciation_alph_dict = {}
        user_misp_output_str = ""
        user_output_words = []

        mispronunciations = []
        new_mispronunciations = []

        if len(ground_truth) > len(predicted):
            return [{"message": "error"}], {}, []

        for i in range(len(ground_truth)):
            ground_truth_word_phonemes = ground_truth[i]
            predicted_word_phonemes = predicted[i]
            original_word = original_text_list[i]
            
            word_mispronounced = False
            
            alignment = self._align_phonemes(ground_truth_word_phonemes, predicted_word_phonemes)

            new_mispronunciations.append(("", ""))

            for item_type, gt_ph, pred_ph in alignment:

                if item_type == 'substitution':
                    word_mispronounced = True
                    mispronunciation_espeak_dict[gt_ph] = mispronunciation_espeak_dict.get(gt_ph, 0) + 1
                    alph_letter = self.phonemes_to_letters(gt_ph)
                    user_misp_output_str = f'You pronounced the "{alph_letter}" sound in "{original_word}" incorrectly (expected "{self.phonemes_to_letters(gt_ph)}", got "{self.phonemes_to_letters(pred_ph)}").\n'
                    
                    if alph_letter in mispronunciation_alph_dict:
                        mispronunciation_alph_dict[alph_letter].append(i)
                    else : 
                        mispronunciation_alph_dict[alph_letter] = [i]

                    mispronunciations.append({
                        "type": "substitution",
                        "message": user_misp_output_str,
                        "index": i
                    })

                    new_mispronunciations[i] = (
                        new_mispronunciations[i][0] + f"-" + self.phonemes_to_letters(gt_ph),
                        new_mispronunciations[i][1] + f"-{item_type}:" + self.phonemes_to_letters(pred_ph)
                    )

                elif item_type == 'deletion':
                    word_mispronounced = True
                    mispronunciation_espeak_dict[gt_ph] = mispronunciation_espeak_dict.get(gt_ph, 0) + 1
                    alph_letter = self.phonemes_to_letters(gt_ph)
                    user_misp_output_str = f'You missed the "{alph_letter}" sound in "{original_word}".\n'

                    if alph_letter in mispronunciation_alph_dict:
                        mispronunciation_alph_dict[alph_letter].append(i)
                    else : 
                        mispronunciation_alph_dict[alph_letter] = [i]

                    mispronunciations.append({
                        "type": "substitution",
                        "message": user_misp_output_str,
                        "index": i
                    })

                    new_mispronunciations[i] = (
                        new_mispronunciations[i][0] + f"-" + self.phonemes_to_letters(gt_ph),
                        new_mispronunciations[i][1] + f"-{item_type}:" + self.phonemes_to_letters(gt_ph)
                    )
                    
                elif item_type == 'insertion':
                    word_mispronounced = True
                    alph_letter = self.phonemes_to_letters(pred_ph)
                    user_misp_output_str = f'You added the "{alph_letter}" sound in "{original_word}".\n'

                    if alph_letter in mispronunciation_alph_dict:
                        mispronunciation_alph_dict[alph_letter].append(i)
                    else : 
                        mispronunciation_alph_dict[alph_letter] = [i]

                    mispronunciations.append({
                        "type": "substitution",
                        "message": user_misp_output_str,
                        "index": i
                    })            
                
                    new_mispronunciations[i] = (
                        new_mispronunciations[i][0] + f"-" + self.phonemes_to_letters(gt_ph),
                        new_mispronunciations[i][1] + f"-{item_type}:" + self.phonemes_to_letters(pred_ph)
                    )
                else:
                    new_mispronunciations[i] = (
                        new_mispronunciations[i][0] + f"-" + self.phonemes_to_letters(gt_ph),
                        new_mispronunciations[i][1] + f"-{item_type}:" + self.phonemes_to_letters(pred_ph)
                    )

        return mispronunciations, mispronunciation_alph_dict, new_mispronunciations
            
        #     if word_mispronounced:
        #         alphabetic_predicted_word = "".join([self.phonemes_to_letters(ph) for ph in predicted_word_phonemes])
        #         user_output_words.append(alphabetic_predicted_word)
        #     else:
        #         user_output_words.append(original_word)
        
        # user_output = " ".join(user_output_words)

        # return mispronunciation_espeak_dict, mispronunciation_alph_dict, user_output, user_misp_output_str

    def run(self, audio, ground_truth_text):
        gt_phonemes, pred_phonemes = self.transcriber.transcribe_audio(
            audio,
            phonemize_gt=True,
            ground_truth_text=ground_truth_text
        )

        # print("Ground Truth (Phonemes):", gt_phonemes)
        # print("Predicted (Phonemes):", pred_phonemes, "\n")

        mispronunciations, mispronunciation_alph_dict, new_mispronunciations = self.find_mispronunciations(gt_phonemes, pred_phonemes, ground_truth_text)

        return mispronunciations, mispronunciation_alph_dict, new_mispronunciations

        # print("User Output:", user_output)
        # print("Mispronunciations:")
        # print(user_misp_output_str)
        # print("Mispronunciation Espeak dictionary:", mispronunciation_espeak_dict)
        # print("Mispronunciation Alphabet dictionary:", mispronunciation_alph_dict)

loader = None
model = None
processor = None
detector = None

def load_md_model():
    global loader
    global model
    global processor
    global detector

    if detector is None:
        loader = LoadModel(MMS_PATH, device=device)
        loader.load_model_and_processor()
        model = loader.get_model()
        processor = loader.get_processor()
        detector = MispronunciationDetection(model, processor)

def run_mispronunciation_detection(audio, ground_truth, i):
    mispronunciations, mispronunciation_alph_dict, new_mispronunciations = detector.run(
        f"{ROOT_PATH}/media/paragraph_{i}.wav",
        ground_truth
    )

    return mispronunciations, mispronunciation_alph_dict, new_mispronunciations
