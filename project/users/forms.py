from django import forms
from django.forms import ModelForm
from smile.models import USER

'''
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
'''

class UserForm(ModelForm):
    PASSWORD = forms.CharField(widget=forms.PasswordInput)
    SEX_CD = forms.CharField(widget=forms.Select)

    class Meta:
        model = USER
        fields = ['EMAIL', 'PASSWORD', 'USER_NM', 'SEX_CD', 'USER_AGE']

