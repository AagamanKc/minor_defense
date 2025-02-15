from django.shortcuts import render
from .forms import ImageUploadForm
from .models import Person
import cv2
import face_recognition
import pickle
import uuid
from django.contrib import messages

# Function to encode face and store it in the database
def encode_and_store_faces(image_path, name):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img)
    
    if encodings:
        encoding = encodings[0]
        # Check if the person already exists, otherwise create a new one
        person, created = Person.objects.get_or_create(name=name)
        
        # Update the unique_id and encoding if the person exists (or create if it's a new entry)
        person.unique_id = person.unique_id or str(uuid.uuid4())  # Ensure unique_id is set if not already
        person.encoding = pickle.dumps(encoding)  # Store the encoding as a binary object
        person.save()  # Save the changes to the person
        
        print(f"Stored encoding for {name} ✅")
        return person
    
    return None


# View to handle the upload and encoding
def upload(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)  # Handle the image upload with request.FILES
        if form.is_valid():
            # Save the form (name and image)
            person = form.save()

            # Get the image path from the saved object
            image_path = person.image.path

            # Call the function to encode and save the encoding
            encode_and_store_faces(image_path, person.name)

            # Show a success message without redirecting
            messages.success(request, "Image and encoding saved successfully!")

            # Re-render the same page with the form and the success message
            return render(request, 'encoding/upload.html', {'form': form})

    else:
        form = ImageUploadForm()

    return render(request, 'encoding/upload.html', {'form': form})
