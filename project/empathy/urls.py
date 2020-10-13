from django.urls import path, include
from empathy import views as empathy_views




urlpatterns = [

        path('1/', empathy_views.training1, name='training1'),
        path('2/', empathy_views.training2, name='training2'),
        path('3/', empathy_views.training3, name='training3'),
        path('4/', empathy_views.training4, name='training4'),
        path('5/', empathy_views.training5, name='training5'),
        path('6/', empathy_views.training6, name='training6'),
        path('7/', empathy_views.training_result, name='training_result'),

        path('1/test', empathy_views.happy_training, name='happy_training'),
        path('2/test', empathy_views.angry_training, name='angry_training'),
        path('3/test', empathy_views.sad_training, name='sad_training'),
        path('4/test', empathy_views.surprise_training, name='surprise_training'),

]