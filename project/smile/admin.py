from django.contrib import admin

# Register your models here.
from smile.models import Phrase, User, Face

admin.site.register(Phrase)
admin.site.register(User)
admin.site.register(Face)