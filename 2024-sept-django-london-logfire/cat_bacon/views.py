from uuid import uuid4

import httpx
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404

from openai import Client

from .forms import ImageForm
from .models import Image

client = Client()


def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file, image_url = create_image(client, form.instance.animal, form.instance.artist)
            print(f'image_file: {image_file!r}, image_url: {image_url!r}')
            form.instance.file_path = image_file
            form.instance.url = image_url
            image = form.save()
            return redirect('image-details', image_id=image.id)
    else:
        form = ImageForm()

    return render(request, 'index.html', {'form': form})


def image_details(request, image_id):
    image = get_object_or_404(Image, id=image_id)
    return render(request, 'image.html', {'image': image})


def create_image(openai_client: Client, animal: str, artist: str) -> tuple[str | None, str]:
    prompt = f'Create an image of a {animal} in the style of {artist}'
    response = openai_client.images.generate(prompt=prompt, model='dall-e-3')

    image_url = response.data[0].url
    # return None, image_url

    r = httpx.get(image_url)
    r.raise_for_status()
    path = f'{uuid4().hex}.jpg'
    (settings.MAIN_STATIC / path).write_bytes(r.content)
    return path, image_url
