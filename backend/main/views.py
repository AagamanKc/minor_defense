import os
import cv2
import pickle
import numpy as np
import face_recognition
import requests
import random
import asyncio
from telegram import Bot
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from encoding.models import Person
from main.silent_face.test import test  # Anti-spoofing model

# ESP32 Camera Stream URL (Use /capture for images, /stream for continuous feed)
ESP32_STREAM_URL = "http://192.168.43.190/capture"  # Adjust this IP

# Set model directory
model_dir = os.path.join(settings.BASE_DIR, "main/silent_face/resources/anti_spoof_models")
device_id = 0  # Change if using GPU

def generate_unique_otp():
    """Generate a unique 4-digit OTP"""
    return str(random.randint(1000, 9999))

your_telegram_bot_token = "7878300541:AAHF2aF3v2cxXkePujiMNsFv3FCPQu0LT5A"
chat_id = 6328640036
async def send_telegram_message(otp, chat_id):
    """Send OTP to a specific Telegram user."""
    try:
        bot = Bot(token=your_telegram_bot_token)
        message = f"Your OTP is: {otp}"
        await bot.send_message(chat_id=chat_id, text=message)  # ✅ Await async function
        print("✅ OTP sent successfully")
    except Exception as e:
        print(f"❌ Error sending OTP: {e}")

def load_encodings():
    """Load all face encodings from the database."""
    persons = Person.objects.all()
    encodeListKnown = []
    studentIds = []
    for person in persons:
        encodeListKnown.append(pickle.loads(person.encoding))  # Decode binary encoding
        studentIds.append(person.name)
    return encodeListKnown, studentIds

@csrf_exempt
def face_recognition_api(request):
    """Continuously capture images from ESP32 and process face recognition."""
    try:
        encodeListKnown, studentIds = load_encodings()  # Load known encodings once
        check_count = 0  # Counter to track the number of face checks

        while True:
            check_count += 1  # Increment count every loop iteration
            # Fetch latest frame from ESP32
            print("connection error if no scanned")
            response = requests.get(ESP32_STREAM_URL)
            print("conncetion established")
            if response.status_code != 200:
                return JsonResponse({"status": "error", "message": "Failed to fetch frame from ESP32 camera."})
            # Convert response bytes to OpenCV image
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue  # Skip if frame couldn't be decoded

            faceCurrFrame = face_recognition.face_locations(frame)
            print("scanned")
            if not faceCurrFrame:
                continue
            # Run anti-spoofing check
            label, value = test(frame, model_dir, device_id)
            if label != 2 or value < 0.5:
                new_otp = generate_unique_otp()
                asyncio.run(send_telegram_message(new_otp, chat_id))
                response_data = {
                    "status": "unreliable picture",
                    "signal": 0,
                    "message": "Fake face detected!",
                    "checks": check_count,
                    "otp": new_otp
                }
                
                print("Sending response:", response_data)  # Debugging log
                return JsonResponse(response_data)

            # Resize and convert image
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Detect faces
            faceCurrFrame = face_recognition.face_locations(imgS)
            encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

            if not faceCurrFrame:
                continue  # No face detected, keep checking

            # Compare detected faces with known encodings
            for encodeFace in encodeCurrFrame:
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    return JsonResponse({"status": "success", "signal": 1, "message": f"✅ Known face: {studentIds[matchIndex]}", "checks": check_count})
                else:
                    new_otp = generate_unique_otp()
                    send_telegram_message(new_otp)
                    return JsonResponse({"status":"unsuccessful","signal":0,"message":"intruder","otp":new_otp})


            # If no known face is found, continue
            continue

    except Exception as e:
        return JsonResponse({"status": "error", "message": f"An error occurred: {str(e)}", "checks": check_count})


'''
Start your Django server:
python manage.py runserver

Send a request from the frontend or a script:
curl -X POST http://127.0.0.1:8000/face_recognition/
'''