from django import forms
from .models import Image

ARTISTS = (
    'Francis Bacon',
    'Edvard Munch',
    'Pablo Picasso',
    'Salvador Dali',
    'Vincent van Gogh',
    'Andy Warhol',
)


class ImageForm(forms.ModelForm):
    animal = forms.CharField(max_length=255)
    artist = forms.ChoiceField(choices=[(artist, artist) for artist in ARTISTS])

    class Meta:
        model = Image
        fields = ['animal', 'artist']
