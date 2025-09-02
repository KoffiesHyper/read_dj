from django.urls import path
from .views import *

urlpatterns = [
    path("", ReadAttemptView),
    path("story-gen/", StoryGenView),
    path("test", TestView)
]