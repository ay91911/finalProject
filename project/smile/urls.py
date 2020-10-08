from django.urls import path, include
from smile import views as smile_views


urlpatterns = [

    path('',smile_views.index, name='index'),
    path('today_phrase_cam/', smile_views.video_today_phrase, name='today_phrase'),
    path('today_phrase_cam_2/',smile_views.video, name='today_phrase_2'),
    path('neutral_cam/', smile_views.video_neutral, name='video_neutral'),
    path('level_1_cam/', smile_views.video_smile_level1, name='video_smile_level1'),
    path('level_2_cam/', smile_views.video_smile_level2, name='video_smile_level2'),
    path('level_3_cam/', smile_views.video_smile_level3, name='video_smile_level3'),
    path('Phrase_list/', smile_views.ListPhrase, name='Phrase_list'),
    path('imageToDB/', smile_views.imageToDB, name='imageToDB'),
    path('neutral_img/',smile_views.img_smile_neutral, name='img_neutral'),
    path('level_1_img/',smile_views.img_smile_level_1, name='img_smile_level1'),
    path('level_2_img/',smile_views.img_smile_level_2, name='img_smile_level2'),
    path('level_3_img/',smile_views.img_smile_level_3, name='img_smile_level3'),

]