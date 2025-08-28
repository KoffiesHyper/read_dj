import subprocess

espeak_phonemes = [
    "a", "aI", "aU", "tS", "T", "S", "Z", "b", "d", "e", "eI", "f", "h", "i:",
    "I", "I@", "j", "k", "l", "m", "n", "N", "oU", "O", "OI", "p", "r", "s", "t",
    "u:", "U", "U@", "V", "3:", "@", "@U", "g"
]

ipa_to_espeak = {}

for code in espeak_phonemes:
    result = subprocess.run(
        ["espeak-ng", "-v", "en", "--ipa", "-x", f"[{code}]"],
        capture_output=True,
        text=True
    )
    ipa = result.stdout.strip()
    ipa_to_espeak[ipa] = code

print(ipa_to_espeak)