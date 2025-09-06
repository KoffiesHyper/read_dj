from transformers import AutoProcessor, AutoModelForCTC
import torch
import torchaudio

processor = AutoProcessor.from_pretrained("aadel4/Wav2vec_Classroom_FT")
model = AutoModelForCTC.from_pretrained("aadel4/Wav2vec_Classroom_FT")

def transcribe_with_class_w2v(waveform):

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription
