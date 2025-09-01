import ffmpeg
import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from utils.kidwhisper import transcribe_waveform_direct
from utils.silero_vad import silero_vad_steam
from utils.conversions import convert_webm_to_wav

from utils.story_generation.InitialParasLinked import run_inital_paras
from utils.story_generation.MatchLinked import run_match
from utils.story_generation.StoryGenLinked import run_story_gen
from utils.story_generation.NoOutlineGenLinked import run_no_outline_gen

CHUNK_THRESHOLD = 5

class LatestTaskRunner:
    def __init__(self):
        self.current_task = None
        self.next_waveform = None

    def add_task(self, waveform, transcribe_fn):
        print("TASK ADDED!")
        self.next_waveform = waveform
        if not self.current_task or self.current_task.done():
            self.current_task = asyncio.create_task(self._runner(transcribe_fn))
    
    async def _runner(self, transcribe_fn):
        while self.next_waveform is not None:
            waveform = self.next_waveform
            self.next_waveform = None

            await transcribe_fn(waveform)

class AudioStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.chunk_buffer = []
        self.last_speaking_time = 0
        self.running_chunks = 0
        self.paragraph = 0
        
        self.task_runner = LatestTaskRunner()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if text_data == "clear":
            print("cleared")
            self.chunk_buffer = []
            self.last_speaking_time = 0
            self.running_chunks = 0
            self.task_runner = LatestTaskRunner()
            self.paragraph = self.paragraph + 1

        elif bytes_data:
            print("received")
            self.chunk_buffer.append(bytes_data)

            if len(self.chunk_buffer) >= 1:
                combined = b"".join(self.chunk_buffer)
                # self.chunk_buffer = []

                speaking = self.run_vad_on_chunk(combined)

                await self.send(text_data=json.dumps({
                    "speaking": speaking
                }))

    def run_vad_on_chunk(self, audio_bytes: bytes, sample_rate=16000):
        try:
            timestamps, waveform = silero_vad_steam(audio_bytes)
            
            if len(timestamps) > 0 and timestamps[-1]["end"] > self.last_speaking_time:
                print("speaking")
                self.last_speaking_time = timestamps[-1]["end"]
                self.running_chunks += 1

                if self.running_chunks >= CHUNK_THRESHOLD:
                    self.running_chunks = 0
                    self.task_runner.add_task(waveform, self.transcribe_and_send)

                return True
            else:
                if self.running_chunks > 0:
                    self.task_runner.add_task(waveform, self.transcribe_and_send)
                
                self.running_chunks = 0
                return False

        except ffmpeg.Error as e:
            print("🔴 FFmpeg decoding error:\n", e.stderr.decode(errors="ignore"))
            return False
    
    async def transcribe_and_send(self, waveform):

        print("TRANSCRIBING AUDIO")
        transcript = transcribe_waveform_direct(waveform, 16000)
        print("DONE")

        await self.send(text_data=json.dumps({
            "transcript": transcript, "paragraph": self.paragraph
        }))


class GenerateStoryConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            mistakes = json.loads(text_data)
            run_inital_paras(mistakes)
            run_match()
            run_story_gen()
            paragraphs = run_no_outline_gen()

            await self.send(text_data=json.dumps(paragraphs))