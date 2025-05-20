# run py -m main.main

import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
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

# Load known encodings from the database
def load_encodings():
    persons = Person.objects.all()
    encodeListKnown = []
    studentIds = []
    for person in persons:
        encodeListKnown.append(pickle.loads(person.encoding))  # Decode binary encoding
        studentIds.append(person.name)
    return encodeListKnown, studentIds


def main():

    encodeListKnown, studentIds = load_encodings()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting live anti-spoofing detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            continue

        # Resize and convert to RGB for face recognition
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faceCurrFrame = face_recognition.face_locations(imgS)
        encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

        if not faceCurrFrame:
            print("No face detected.")
            cv2.imshow("Face Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            # Extract the face region from the original image
            top, right, bottom, left = [v * 4 for v in faceLoc]  # Scale back up
            face_img = frame[top:bottom, left:right]  # Crop the detected face

            # Run anti-spoofing detection
            label, value = test(frame, model_dir, device_id)
            print(f"label= {label}, value= {value}")

            name = "Spoofing"  # Default name

            if label == 1 and value > 0.6:  # Only accept real faces
                print("Real face detected, proceeding with recognition.")

                # Run face recognition
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = studentIds[matchIndex]
                    print(f"Known face detected: {name}")
                else:
                    name = "Unknown"
                    print("Unknown face detected.")
            else:
                print("Spoofing detected or confidence too low! Ignoring face.")

            # Draw bounding box and name
            color = (0, 255, 0) if name != "Spoofing" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y_offset = 10  # How much above the box you want the name
            cv2.rectangle(frame, (left, top - 35 - y_offset), (right, top - y_offset), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, top - 10 - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


        # Show the video frame with results
        cv2.imshow("Face Attendance", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        '''try:
            label, value = test(frame, model_dir, device_id)
            text = f"{'Real' if label == 1 else 'Fake'} Face | Score: {value:.2f}"
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception as e:
            print(f"Error: {e}")
            cv2.putText(frame, "Error processing frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Live Anti-Spoofing', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()'''

if __name__ == "__main__":
    main()
