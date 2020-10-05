from django.shortcuts import render, redirect
from django.contrib import messages

def mainpage(request):
    return render(request, 'service/mainpage1.html')


def smile_study(request):
    #return render(request, 'smile/index.html')
    return render(request, 'smile/todayPhrase.html')

def empathy_training(request):
    return render(request, 'empathy/training.html')