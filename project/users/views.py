from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserForm

def register(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            USER_NM = form.cleaned_data.get('USER_NM')
            messages.success(request, f'Account created for {USER_NM}!')
            return redirect('service-mainpage1')
    else:
        form = UserForm()
    return render(request, 'users/register.html', {'form': form})
