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
    face_best = [face_list.SMILE1_PERCENT, face_list.SMILE2_PERCENT, face_list.SMILE3_PERCENT]
    smile_dic = {face_list.SMILE1_PATH:face_list.SMILE1_PERCENT,
                 face_list.SMILE2_PATH:face_list.SMILE2_PERCENT,
                 face_list.SMILE3_PATH:face_list.SMILE3_PERCENT}

    percent_max = 0
    for i in face_best:
        if i > percent_max:
            percent_max = i
    print(percent_max)

    for keys, values in smile_dic.items():
        if values == percent_max:
            best_smile = keys

    # user = USER.objects.get(pk=request.session["userEmail"])
    context = {
        # 'posts': FACE.filter(EMAIL=user).objects.all(),
        'faces': face_list,
        'percent' : percent_max,
        'best_smile':best_smile,

    }

    return render(request, 'status/compare.html', context)