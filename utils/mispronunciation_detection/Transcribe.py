import os
import torch
import librosa
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import numpy as np
import re
import subprocess

# Set eSpeak NG library path (adjust if needed)
ESPEAK_DLL_PATH = "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib"
if os.path.exists(ESPEAK_DLL_PATH):
    EspeakWrapper.set_library(ESPEAK_DLL_PATH)
    print(f"eSpeak NG library path set to: {ESPEAK_DLL_PATH}")
else:
    print(f"Warning: eSpeak NG DLL not found at {ESPEAK_DLL_PATH}.")

class Transcribe:
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
        """
        Takes a preloaded model and processor.
        """
        self.model = model
        self.processor = processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _sorted_phonemes(self):
        return sorted(self.PHONEME_MAP.keys(), key=len, reverse=True)

    def _split_phoneme_string(self, phoneme_string: str):
        """Splits a phoneme string into a list of individual phoneme tokens."""
        sorted_phs = self._sorted_phonemes()
        # Create a regex pattern to match the longest phonemes first
        pattern = '|'.join(re.escape(p) for p in sorted_phs)
        return [m.group(0) for m in re.finditer(pattern, phoneme_string)]

    def transcribe_audio(self, audio, phonemize_gt=False, ground_truth_text=None, phonemizer_lang="en-us"):
        """
        Transcribes a single audio file. Optionally phonemizes ground truth.
        """
        # Load and normalize audio
        # sr = 16000
        audio, sr = librosa.load(audio, sr=16000)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)

        # Prepare input
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_phonemes_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        # Process pred_phonemes into a list of lists of individual phonemes
        pred_phonemes_list_of_lists = []
        # The model's output is a string of phonemes for the entire utterance.
        # We need to split this into words and then into individual phonemes.
        # Assuming spaces in pred_phonemes_str delineate word boundaries.
        for phoneme_word_str in pred_phonemes_str.split():
            pred_phonemes_list_of_lists.append(self._split_phoneme_string(phoneme_word_str))

        # Optionally phonemize ground truth
        gt_phonemes_list_of_lists = None
        if phonemize_gt and ground_truth_text:
            gt_phonemes_list_of_lists = []
            # Phonemize the entire ground truth text at once
            # This returns a list of strings, where each string is the phonemes for a word
            phonemized_words_raw = phonemize(ground_truth_text.split(), language=phonemizer_lang, backend='espeak',
                                             strip=True, preserve_punctuation=False)
            for phoneme_word_str in phonemized_words_raw:
                gt_phonemes_list_of_lists.append(self._split_phoneme_string(phoneme_word_str))

        return gt_phonemes_list_of_lists, pred_phonemes_list_of_lists
