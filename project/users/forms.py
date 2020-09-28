
from django import forms
from django.forms import ModelForm
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from smile.models import User


GENDER=[
    ('male', 'Male')
    ('female', 'Female')
]

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()
    gender = forms.CharField(label='Gender', widget=forms.Select(choices=GENDER))

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'gender']


GENDER=[
    ('male', 'Male')
    ('female', 'Female')
]


class UserForm(ModelForm):
    Password = forms.CharField(widget=forms.PasswordInput)
    Gender = forms.CharField(widget=forms.Select)

    class Meta:
        model = User
        fields = ['Email', 'Password', 'Username', 'Gender', 'Age']
