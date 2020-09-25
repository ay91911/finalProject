from django.urls import path, include
from smile import views as smile_views


urlpatterns = [
    path('video/', smile_views.video_smile, name='Video_smile'),
    path('video/1/',smile_views.video_none, name='Video_none'),

]