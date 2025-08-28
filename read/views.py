from rest_framework.decorators import api_view
from rest_framework.status import HTTP_200_OK
from rest_framework.response import Response

import torch
import torchaudio
import numpy as np

from utils.compare import compare_strings, check_missing_words
from utils.kidwhisper import transcribe_waveform_direct
from utils.silero_vad import silero_vad
from utils.voice_type_classifier import voice_type_classifier
from utils.mispronunciation_detection.mispronunciation_detection import run_mispronunciation_detection

@api_view(["GET"])
def TestView(request):
    return Response("Working!")

@api_view(["POST"])
def ReadAttemptView(request):
    recording = request.FILES["recording"]
    audio_bytes = recording.read()   

    speech_timestamps, waveform, sample_rate = silero_vad(audio_bytes)
    torchaudio.save(f"audio.wav", waveform, sample_rate)
    
    sliced_audio = []

    num_samples = waveform.shape[1]
    duration = num_samples / sample_rate

    for i, timestamp in enumerate(speech_timestamps):
        start_frame = int(timestamp["start"])
        end_frame = int(timestamp["end"])
        sliced_audio.append(waveform[:, start_frame:end_frame])
    
    sliced_audio = torch.cat(sliced_audio, dim=1)
    torchaudio.save(f"sliced.wav", sliced_audio, sample_rate)

    waveform = voice_type_classifier()

    num_samples = waveform.shape[1]
    spoken_duration = num_samples / sample_rate

    transcript = transcribe_waveform_direct(waveform, sample_rate).strip().replace(",", ", ")

    story = request.data.get("story")

    result, accuracy = compare_strings(story.lower(), transcript.lower())

    if result[-1]["text"] == "":
        result = result[:-1]

    missing_words = check_missing_words(story, transcript)

    mispronunciations = []
    if missing_words == 0:
        audio = waveform.numpy()
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
        
        audio = audio.astype(np.float32)
        mispronunciations = run_mispronunciation_detection(waveform, story)

    response = {
        "result": result,
        "stats": {
            "accuracy": accuracy,
            "duration": duration,
            "spoken_duration": spoken_duration
        },
        "mispronunciations": mispronunciations,
        "missing_words": missing_words
    }
    
    return Response(response, status=HTTP_200_OK)