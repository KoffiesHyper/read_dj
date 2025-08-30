from django.apps import AppConfig
from decouple import config
import os

class ReadConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'read'

    def ready(self):
        from utils.kidwhisper import load_model
        from utils.mispronunciation_detection.mispronunciation_detection import load_md_model

        load_model()
        load_md_model()

        device = config("device")

        if device == "cuda":
            home = os.path.expanduser("~")
            os.environ["PATH"] = f"{home}/bin:{home}/sox_build/bin:" + os.environ.get("PATH", "")