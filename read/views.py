from rest_framework.decorators import api_view
from rest_framework.status import HTTP_200_OK
from rest_framework.response import Response

import json
import torch
import torchaudio
import numpy as np

from utils.compare import compare_strings, check_missing_words
from utils.kidwhisper import transcribe_waveform_direct
from utils.silero_vad import silero_vad
from utils.voice_type_classifier import voice_type_classifier
from utils.mispronunciation_detection.mispronunciation_detection import run_mispronunciation_detection
from utils.load_into_paragraphs import load_into_paragraphs
from utils.story_generation.InitialParasLinked import run_inital_paras
from utils.story_generation.MatchLinked import run_match
from utils.story_generation.StoryGenLinked import run_story_gen
from utils.story_generation.NoOutlineGenLinked import run_no_outline_gen

from decouple import config

ROOT_PATH = config("ROOT_PATH")

@api_view(["GET"])
def TestView(request):
    return Response("Working!")

@api_view(["POST"])
def ReadAttemptView(request):
    recording = request.FILES["recording"]
    audio_bytes = recording.read()   

    story = json.loads(request.data.get("story"))
    time_stamps = json.loads(request.data.get("time_stamps"))

    paragraphs, sample_rate = load_into_paragraphs(audio_bytes, time_stamps)

    num_samples = 0
    for paragraph in paragraphs:
        if paragraph == "empty": continue

        num_samples += paragraph.shape[1]
    duration = num_samples / sample_rate

    empty = silero_vad(paragraphs, sample_rate)
    paragraphs = voice_type_classifier(empty)

    num_samples = 0
    for paragraph in paragraphs:
        if paragraph == "empty": continue

        num_samples += paragraph.shape[1]
    spoken_duration = num_samples / sample_rate

    transcripts = transcribe_waveform_direct(paragraphs, sample_rate)
    results, accuracy = compare_strings(story, transcripts)

    missing_words = check_missing_words(story, transcripts)

    mispronunciations = []

    total_mistakes = {}
    mistakes_per_paragraph = []

    for i, missing in enumerate(missing_words):
        print(f"MP: #{i+1}")
        if missing == 0:
            audio = paragraphs[i].numpy()
            if audio.shape[0] > 1:
                audio = np.mean(audio, axis=0)
            
            audio = audio.astype(np.float32)
            mp, mistakes = run_mispronunciation_detection(paragraphs[i], story[i])

            print(mistakes)

            for key in mistakes:
                for mpp in mistakes[key]:
                    mistakes_per_paragraph.append([key, mpp, i])

                if key in total_mistakes:
                    total_mistakes[key] += len(mistakes[key])
                else:
                    total_mistakes[key] = len(mistakes[key])

            mispronunciations.append(mp)
        else:
            mispronunciations.append([])

    response = {
        "result": results,
        "stats": {
            "accuracy": accuracy,
            "duration": duration,
            "spoken_duration": spoken_duration
        },
        "mispronunciations": mispronunciations,
        "mistakes": total_mistakes,
        "mistakes_per_paragraph": mistakes_per_paragraph,
        "missing_words": missing_words
    }
    
    return Response(response, status=HTTP_200_OK)

@api_view(["POST"])
def StoryGenView(request):
    mistakes = request.data.get("mistakes")
    print(mistakes)

    run_inital_paras(mistakes)
    run_match()
    run_story_gen()
    paragraphs = run_no_outline_gen()

    return Response(paragraphs)