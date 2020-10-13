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
    face_list = FACE.objects.filter(EMAIL=request.session["userEmail"]).latest('STUDY_DATE')

    # user = USER.objects.get(pk=request.session["userEmail"])
    context = {
        # 'posts': FACE.filter(EMAIL=user).objects.all(),
        'faces': face_list

    }

    return render(request, 'status/compare.html', context)