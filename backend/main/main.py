# run py -m main.main


import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.backend.settings')
django.setup()
import sys
import time
import pickle
import cv2
import numpy as np
import face_recognition
from django.conf import settings
from main.silent_face.test import test  # Use absolute import
from encoding.models import Person  # Import Person model
print(Person.objects.all())

# Add the silent_face directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from silent_face.test import test  # Import anti-spoofing function

# Set the model directory and device
model_dir = os.path.join(settings.BASE_DIR, "main/silent_face/resources/anti_spoof_models")
device_id = 0  # Change if using GPU

# Start video capture (for testing on a laptop)
cap = cv2.VideoCapture(0)

# Load known encodings from the database
def load_encodings():
    persons = Person.objects.all()
    encodeListKnown = []
    studentIds = []
    for person in persons:
        encodeListKnown.append(pickle.loads(person.encoding))  # Decode binary encoding
        studentIds.append(person.unique_id)
    return encodeListKnown, studentIds

encodeListKnown, studentIds = load_encodings()

# Limit detection speed (process once per second)
last_processed_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        continue  # Skip frame if capture failed

    # Limit the processing speed
    if time.time() - last_processed_time < 1.0:  # 1-second interval
        cv2.imshow("Face Attendance", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    last_processed_time = time.time()  # Update last processed time

    # Resize and convert to RGB for face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    if not faceCurrFrame:
        print("No face detected.")
        cv2.imshow("Face Attendance", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        # Extract the face region from the original image
        top, right, bottom, left = [v * 4 for v in faceLoc]  # Scale back up
        face_img = img[top:bottom, left:right]  # Crop the detected face

        # Run anti-spoofing detection
        label, value = test(face_img, model_dir, device_id)
        if label == 2 and value > 0.99:  # Only accept real faces
            print("Real face detected, proceeding with recognition.")

            # Run face recognition
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                print(f"Known face detected: {studentIds[matchIndex]}")
            else:
                print("Unknown face detected.")
        else:
            print("Spoofing detected or confidence too low! Ignoring face.")


    # Show the video frame with results
    cv2.imshow("Face Attendance", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
