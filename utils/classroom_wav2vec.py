from transformers import AutoProcessor, AutoModelForCTC
import torch
import torchaudio

processor = AutoProcessor.from_pretrained("aadel4/Wav2vec_Classroom_FT")
model = AutoModelForCTC.from_pretrained("aadel4/Wav2vec_Classroom_FT")

def speech_to_text():

    waveform, sample_rate = torchaudio.load("sliced.wav")

    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000

    waveform = waveform.mean(dim=0)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription
