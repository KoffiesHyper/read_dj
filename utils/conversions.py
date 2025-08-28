import io
import ffmpeg
import torchaudio

def convert_webm_to_wav(audio_bytes, sample_rate=16000):
    out, err = (
        ffmpeg
        .input('pipe:0', format='webm')
        .output('pipe:1', format='wav', ac=1, ar=sample_rate)
        .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
    )

    waveform, sr = torchaudio.load(io.BytesIO(out))
    return waveform, sr