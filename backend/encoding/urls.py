
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import upload

urlpatterns = [
    path('upload/', upload, name='upload'),
]

# Add this to serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
