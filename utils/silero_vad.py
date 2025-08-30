from silero_vad import load_silero_vad, get_speech_timestamps
from .conversions import convert_webm_to_wav

import torch
import torchaudio

model = load_silero_vad()

def silero_vad(waveforms, sample_rate):

    for i, waveform in enumerate(waveforms):
        print(f"Silero VAD: Paragraph #{i+1}")
        
        mono = waveform[0].numpy()
        speech_timestamps = get_speech_timestamps(mono, model, sampling_rate=sample_rate)

        sliced_audio = []

        num_samples = waveform.shape[1]
        duration = num_samples / sample_rate

        for j, timestamp in enumerate(speech_timestamps):
            start_frame = int(timestamp["start"])
            end_frame = int(timestamp["end"])
            sliced_audio.append(waveform[:, start_frame:end_frame])
        
        sliced_audio = torch.cat(sliced_audio, dim=1)
        torchaudio.save(f"paragraph_{i}.wav", sliced_audio, sample_rate)
    
def silero_vad_steam(audio_bytes):
    waveform, sample_rate = convert_webm_to_wav(audio_bytes)
    mono = waveform[0].numpy()
    speech_timestamps = get_speech_timestamps(mono, model, sampling_rate=sample_rate)
    return speech_timestamps, waveform