from django.urls import path, include
from smile import views as smile_views


urlpatterns = [

    path('',smile_views.todayPhrase),
    path('test/',smile_views.smileApp),
    #path('2/',smile_views.index02),
    #path('3/',smile_views.index03),
    path('today_phrase/', smile_views.video_today_frame, name='todayPhrase'),
    path('level_1_cam/', smile_views.video_smile_level1, name='video_smile_level1'),
    path('level_2_cam/', smile_views.video_smile_level2, name='video_smile_level2'),
    path('level_3_cam/', smile_views.video_smile_level3, name='video_smile_level3'),

]