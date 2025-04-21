from django.contrib import admin

from .models import Image


class ImageAdmin(admin.ModelAdmin):
    list_display = '__str__', 'timestamp'
    ordering = ('-timestamp',)

    fields = 'animal', 'artist', 'timestamp', 'url', 'file_path'
    readonly_fields = 'animal', 'artist', 'timestamp', 'url', 'file_path'


admin.site.register(Image, ImageAdmin)
