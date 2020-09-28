from django.contrib import admin

# Register your models here.
from smile.models import COM_CD_M,COM_CD_D, PHRASE, USER, FACE

admin.site.register(COM_CD_M)
admin.site.register(COM_CD_D)
admin.site.register(PHRASE)
admin.site.register(USER)
admin.site.register(FACE)