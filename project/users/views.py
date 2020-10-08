from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserForm, LoginForm
from smile.models import USER

def register(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            USER_NM = form.cleaned_data.get('USER_NM')
            messages.success(request, f'Account created for {USER_NM}!')
            return redirect('login')
    else:
        form = UserForm()
    return render(request, 'users/register.html', {'form': form})


def login(request):
    # if request.method == 'POST':
    # form = LoginForm(request.POST)
    #     if form.is_valid():
    #         request.session['EMAIL'] = form.EMAIL
    #         return redirect('mainpage')
    # else:
    form = LoginForm()
    return render(request, 'users/login.html', {'form': form})


def loginProcess(request):
    form = LoginForm(request.POST)
    # if request.method == 'POST':
    if form.is_valid():
        EMAIL = form.cleaned_data["EMAIL"]
        PASSWORD = form.cleaned_data["PASSWORD"]
        print(EMAIL,PASSWORD)

        try:
            user = USER.objects.get(pk=EMAIL, PASSWORD=PASSWORD)
            print(user)
            if user :
                request.session["loginuser"] = user.USER_NM
                request.session["userEmail"] = user.EMAIL
                return redirect('mainpage')
            else :
                errormessage = "1 로그인 실패. 다시 로그인하세요"
                context = {"errormessage": errormessage}
                return render(request, "login.html", context)
        except(USER.DoesNotExist) :
            print("333")
            errormessage="2 로그인 실패. 다시 로그인하세요"
            context = {"errormessage": errormessage}
            return render(request, "login.html", context)

    return render(request, 'users/login.html', {'form': form})