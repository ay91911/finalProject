from django.urls import path, include
from smile import views as smile_views


urlpatterns = [
    path('level_1/', smile_views.video_smile_level1, name='video_smile_level1'),
    path('level_2/', smile_views.video_smile_level2, name='video_smile_level2'),
    path('level_3/', smile_views.video_smile_level3, name='video_smile_level3'),
]