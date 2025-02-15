import os
import cv2
import pickle
import numpy as np
import face_recognition
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from encoding.models import Person
from main.silent_face.test import test  # Anti-spoofing model

# ESP32 Camera Stream URL (change accordingly)
ESP32_STREAM_URL = "http://192.168.1.100:81/stream"  # Adjust this IP

# Set model directory
model_dir = os.path.join(settings.BASE_DIR, "main/silent_face/resources/anti_spoof_models")
device_id = 0  # Change if using GPU

def load_encodings():
    """Load all face encodings from the database."""
    persons = Person.objects.all()
    encodeListKnown = []
    studentIds = []
    for person in persons:
        encodeListKnown.append(pickle.loads(person.encoding))  # Decode binary encoding
        studentIds.append(person.unique_id)
    return encodeListKnown, studentIds

@csrf_exempt
def face_recognition_api(request):
    """Process video stream from ESP32 and return real (1) or fake (0) detection."""
    
    cap = cv2.VideoCapture(ESP32_STREAM_URL)  # Capture video stream
    if not cap.isOpened():
        return JsonResponse({"status": "error", "message": "Failed to connect to ESP32 camera."})

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if no frame is received

        # Run anti-spoofing check
        label, value = test(frame, model_dir, device_id)
        if label != 2 or value < 0.99:
            cap.release()
            return JsonResponse({"status": "spoof_detected", "signal": 0, "message": "❌ Fake face detected!"})

        # Resize and convert image
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces
        faceCurrFrame = face_recognition.face_locations(imgS)
        encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

        if not faceCurrFrame:
            continue  # No face detected, continue checking next frame

        encodeListKnown, studentIds = load_encodings()

        for encodeFace in encodeCurrFrame:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                cap.release()
                return JsonResponse({"status": "success", "signal": 1, "message": f"✅ Known face: {studentIds[matchIndex]}"})

        # If no known face is found
        cap.release()
        return JsonResponse({"status": "unknown", "signal": 0, "message": "Unknown face detected!"})




'''
Start your Django server:
python manage.py runserver

Send a request from the frontend or a script:
curl -X POST http://127.0.0.1:8000/face_recognition/
'''