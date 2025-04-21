from django.db import models


class Image(models.Model):
    id = models.AutoField(primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    animal = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    url = models.URLField(null=True, max_length=2000)
    file_path = models.TextField(null=True)

    def __str__(self):
        return f'{self.animal} in the style of {self.artist}'
