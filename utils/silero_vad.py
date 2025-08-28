from silero_vad import load_silero_vad, get_speech_timestamps
from .conversions import convert_webm_to_wav

model = load_silero_vad()

def silero_vad(audio_bytes):
    waveform, sample_rate = convert_webm_to_wav(audio_bytes)
    mono = waveform[0].numpy()
    speech_timestamps = get_speech_timestamps(mono, model, sampling_rate=sample_rate)
    
    return speech_timestamps, waveform, sample_rate