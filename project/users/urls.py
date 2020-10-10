
from django.urls import path, include
from users import views as user_views

urlpatterns = [
    path('', user_views.login, name='login'),
    path('loginProcess', user_views.loginProcess, name='loginProcess'),
    path('logout', user_views.logout, name='logout'),

]