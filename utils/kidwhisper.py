import os
import certifi
import whisper
import numpy
import torchaudio

from .classroom_wav2vec import transcribe_with_class_w2v

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

def transcribe_waveform_direct(paragraphs, sample_rate, environ_type):
    transcripts = []

    for i, waveform in enumerate(paragraphs):
        print(f"Whisper: Paragraph #{i+1}")

        if waveform == "empty":
            transcripts.append("empty")
            continue

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        if environ_type == "Noisy":
            transcripts.append(transcribe_with_class_w2v(waveform))
        else:
            audio = waveform.squeeze().numpy().astype("float32")

            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))

            result = model.transcribe(audio)["text"].strip().replace(",", ", ")
            transcripts.append(result)

    return transcripts