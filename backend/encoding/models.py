from django.db import models
import uuid

# Create your models here.
class Person(models.Model):
    name = models.CharField(max_length=255,null= False, blank= False)
    unique_id = models.CharField(max_length=36, unique=True, default=uuid.uuid4, editable=False)  # UUIDField is better in newer Django versions
    encoding = models.BinaryField()
    image = models.ImageField(upload_to="faces/", null=False, blank=False)

    def __str__(self):
        return self.name