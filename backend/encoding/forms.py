from django import forms
from .models import Person

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Person
        fields = ["name", "image"]