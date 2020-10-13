from django.shortcuts import render
from smile.models import PHRASE, FACE, USER

# Create your views here.


def recent_smile(request):
    user = USER.objects.get(pk=request.session["userEmail"])
    context = {
        # 'posts': FACE.filter(EMAIL=user).objects.all(),
        'faces': FACE.objects.filter(EMAIL=user),
    }
    return render(request, 'status/compare.html', context)