from rest_framework.decorators import api_view
from rest_framework.status import HTTP_200_OK
from rest_framework.response import Response

import json
import torch
import torchaudio
import numpy as np
import time

from utils.compare import compare_strings, check_missing_words, normalize_text
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
    start = time.time()

    recording = request.FILES["recording"]
    audio_bytes = recording.read()   

    story = json.loads(request.data.get("story"))
    time_stamps = json.loads(request.data.get("time_stamps"))

    voice_type = request.data.get("voice_type")
    environ_type = request.data.get("environ_type")

    cur_paragraph = int(request.data.get("paragraph"))

    paragraphs, sample_rate = load_into_paragraphs(audio_bytes, time_stamps)

    num_samples = 0
    for p in paragraphs:
        if p == "empty": continue

        num_samples += p.shape[1]
    duration = num_samples / sample_rate

    sv_start = time.time()
    empty = silero_vad(paragraphs, sample_rate, cur_paragraph)
    sv_end = time.time()

    vtc_start = time.time()
    paragraphs = voice_type_classifier(empty, voice_type, cur_paragraph)
    vtc_end = time.time()

    num_samples = 0
    for paragraph in paragraphs:
        if paragraph == "empty": continue

        num_samples += paragraph.shape[1]
    spoken_duration = num_samples / sample_rate

    t_start = time.time()
    transcripts = transcribe_waveform_direct(paragraphs, sample_rate, environ_type)
    t_end = time.time()
    results, accuracy = compare_strings(story, transcripts)

    missing_words = check_missing_words(story, transcripts)

    mispronunciations = []

    total_mistakes = {}
    mistakes_per_paragraph = []

    new_mis = []

    mp_start = time.time()
    for i, missing in enumerate(missing_words):
        print(f"MP: #{i+1}")
        if missing == 0:
            audio = paragraphs[i].numpy()
            if audio.shape[0] > 1:
                audio = np.mean(audio, axis=0)
            
            audio = audio.astype(np.float32)

            mp, mistakes, new_mispronunciations = run_mispronunciation_detection(paragraphs[i], " ".join(normalize_text(story[i])), cur_paragraph)
            new_mis.append(new_mispronunciations)
            for key in mistakes:
                for mpp in mistakes[key]:
                    mistakes_per_paragraph.append([key, mpp, cur_paragraph])

                if key in total_mistakes:
                    total_mistakes[key] += len(mistakes[key])
                else:
                    total_mistakes[key] = len(mistakes[key])

            mispronunciations.append(mp)
        else:
            mispronunciations.append([])

    mp_end = time.time()
    end = time.time()

    if len(new_mis) == 0:
        new_mis.append([])

    with open("performance.txt", "a") as f:
        if missing_words[0] == 0:
            f.write(f"{end-start},{sv_end-sv_start},{vtc_end-vtc_start},{t_end-t_start}, {mp_end-mp_start}\n")

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
        "missing_words": missing_words,
        "audio": [request.build_absolute_uri(f"/media/paragraph_{i}.wav") for i in range(7)],
        "new_mispronunciations": new_mis
    }
    
    return Response(response, status=HTTP_200_OK)

@api_view(["POST"])
def StoryGenView(request):
    mistakes = request.data.get("mistakes")

    run_inital_paras(mistakes)
    run_match()
    run_story_gen()
    paragraphs = run_no_outline_gen()

    return Response(paragraphs)