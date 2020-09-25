from django.shortcuts import render, redirect
from django.contrib import messages

def mainpage(request):
    return render(request, 'service/mainpage1.html')
