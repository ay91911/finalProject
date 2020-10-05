from django.urls import path, include
from empathy import views as empathy_views


urlpatterns = [

    path('level_1_cam/', empathy_views.video_smile_level1, name='video_smile_level1'),

]