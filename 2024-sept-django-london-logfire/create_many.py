import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'logfire_django_demo.settings')

django.setup()

from cat_bacon.models import Image

# create 10_000 images
images = [Image(animal='cat', artist='Francis Bacon', url='https://cataas.com/cat') for _ in range(10_000)]
Image.objects.bulk_create(images)
