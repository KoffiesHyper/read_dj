import subprocess
import torch
import torchaudio

env_name = "pyannote"
wav_path = "/Users/koffieshyper/Desktop/read_dj/sliced.wav"
script_path = "/Users/koffieshyper/Desktop/read_dj/voice_type_classifier/apply.sh"
command = f"source /opt/anaconda3/etc/profile.d/conda.sh && conda init && conda activate {env_name} && {script_path} {wav_path}"

target_category = "MAL"

def voice_type_classifier():
    
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        executable="/bin/bash"
    )

    segments = []

    with open("/Users/koffieshyper/Desktop/read_dj/output_voice_type_classifier/sliced/all.rttm") as file:
        for line in file:
            line = line.strip().split(" ")
            category = line[7]

            if category == target_category:
                segments.append([float(line[3]), float(line[4])])

    waveform, sample_rate = torchaudio.load("sliced.wav")
    sliced_audio = []

    num_samples = waveform.shape[1]
    duration = num_samples / sample_rate

    for i, timestamp in enumerate(segments):
        start_frame = int(timestamp[0] * sample_rate)
        end_frame = int(timestamp[1] * sample_rate)
        sliced_audio.append(waveform[:, start_frame:end_frame])
    
    sliced_audio = torch.cat(sliced_audio, dim=1)
    
    return sliced_audio