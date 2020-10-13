from django.urls import path, include
from status import views as status_views

urlpatterns = [
    path('compare_01/',status_views.recent_smile, name='recent_smile'),


]