from django.shortcuts import render, redirect
from django.contrib import messages
from smile.models import PHRASE, FACE, USER

def mainpage(request):
    return render(request, 'service/mainpage1.html')

def smile_prepare(request):
    return render(request, 'smile/start.html')

def smile_study(request):
    return render(request, 'smile/emotion_detection.html')

def empathy_training(request):
    return render(request, 'empathy/training.html')

def compare_photos(request):
    return render(request, 'status/compare.html')