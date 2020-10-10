"""project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from service import views as service_views
from smile import views as smile_views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', service_views.mainpage, name='mainpage'),
    path('', service_views.mainpage, name='mainpage'),
    path('', service_views.mainpage, name='mainpage'),
    path('smile0/', service_views.smile_prepare, name='smile_prepare'),
    path('smile/', service_views.smile_study, name='smile_study'),
    path('empathy/', service_views.empathy_training, name='empathy_training'),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
