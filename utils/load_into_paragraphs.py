from .conversions import convert_webm_to_wav

def load_into_paragraphs(audio_bytes, time_stamps):
    waveform, sample_rate = convert_webm_to_wav(audio_bytes)

    timestamps_samples = [int((t / 1000) * sample_rate) for t in time_stamps]

    paragraphs = []
    for i in range(len(timestamps_samples) - 1):
        start = timestamps_samples[i]
        end = timestamps_samples[i + 1]
        paragraphs.append(waveform[..., start:end])
    
    return paragraphs, sample_rate