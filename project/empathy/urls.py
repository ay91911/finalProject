from django.urls import path, include
from empathy import views as empathy_views


urlpatterns = [

    path('level_1_cam/', empathy_views.video_smile_level1, name='video_smile_level1'),

    path('1/', empathy_views.training1, name='training1'),
    path('2/', empathy_views.training2, name='training2'),
    path('3/', empathy_views.training3, name='training3'),
    path('4/', empathy_views.training4, name='training4'),

]