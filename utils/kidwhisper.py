import os
import certifi
import whisper
import numpy
import torchaudio

os.environ['SSL_CERT_FILE'] = certifi.where()

model = None

def load_model():
    global model

    if model is None:
        model = whisper.load_model("small")
    return model

def transcribe_with_whisper(wav_path: str) -> str:
    result = model.transcribe(wav_path)
    return result['text']

def transcribe_waveform_direct(waveform, sample_rate):
    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Convert to float32 NumPy array
    audio = waveform.squeeze().numpy().astype("float32")

    # Normalize to range -1.0 to 1.0 if not already
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / max(abs(audio.max()), abs(audio.min()))

    result = model.transcribe(audio)
    return result["text"]
