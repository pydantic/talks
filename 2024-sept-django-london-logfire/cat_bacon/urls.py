from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('image/<int:image_id>/', views.image_details, name='image-details'),
]
