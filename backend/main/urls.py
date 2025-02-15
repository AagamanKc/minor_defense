from django.urls import path
from .views import face_recognition_api

urlpatterns = [
    path("recognize/", face_recognition_api, name="face-recognition"),
]