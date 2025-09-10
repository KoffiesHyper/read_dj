import subprocess
import torch
import torchaudio

from decouple import config

ROOT_PATH = config("ROOT_PATH")
VTC_PATH = config("VTC_PATH")
CONDA_PATH = config("CONDA_PATH")

env_name = "pyannote"
wav_path = f"{ROOT_PATH}/media"
script_path = f"{VTC_PATH}/apply.sh"
command = f"source {CONDA_PATH}/etc/profile.d/conda.sh && conda init && conda activate {env_name} && {script_path} {wav_path}"

categories = {
    "Male": "MAL",
    "Female": "FEM",
    "Child": "KCHI"
}

def get_ith_command(i):
    return f"source {CONDA_PATH}/etc/profile.d/conda.sh && conda init && conda activate {env_name} && {script_path} {wav_path}/paragraph_{i}.wav"

def voice_type_classifier(empty, voice_type, cur_paragraph):
    paragraphs = []

    for i in range(1):            
        print(f"VTC: Paragraph #{i+1}")

        if i in empty:
            paragraphs.append("empty")
            continue

        command = get_ith_command(cur_paragraph)

        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        print("STDOUT:")
        print(result.stdout)

        print("STDERR:")
        print(result.stderr)

        segments = []

        with open(f"{ROOT_PATH}/output_voice_type_classifier/paragraph_{cur_paragraph}/all.rttm") as file:
            for line in file:
                line = line.strip().split(" ")
                category = line[7]

                if category == categories[voice_type]:
                    segments.append([float(line[3]), float(line[4])])

        waveform, sample_rate = torchaudio.load(f"{ROOT_PATH}/media/paragraph_{cur_paragraph}.wav")
        sliced_audio = []

        num_samples = waveform.shape[1]
        duration = num_samples / sample_rate

        for i, timestamp in enumerate(segments):
            start_frame = int(timestamp[0] * sample_rate)
            end_frame = int(timestamp[1] * sample_rate)
            sliced_audio.append(waveform[:, start_frame:end_frame])
        

        if len(sliced_audio) == 0:
            sliced_audio = torch.zeros((1, 1))
            if i in empty:
                paragraphs.append("empty")
        else:
            sliced_audio = torch.cat(sliced_audio, dim=1)

        paragraphs.append(sliced_audio)
        
    return paragraphs