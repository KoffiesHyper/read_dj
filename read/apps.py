from django.apps import AppConfig


class ReadConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'read'

    def ready(self):
        from utils.kidwhisper import load_model
        from utils.mispronunciation_detection.mispronunciation_detection import load_md_model

        load_model()
        load_md_model()