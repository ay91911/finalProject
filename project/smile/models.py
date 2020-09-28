from datetime import datetime

from django.db import models

class COM_CD_M(models.Model):
    COM_CD = models.CharField(max_length=20, primary_key=True)
    COM_CD_NM = models.CharField(max_length=20)
    REMARK_DC = models.CharField(max_length=20)
    USE_YN = models.CharField(max_length=20)

    def __str__(self):
        return self.COM_CD

class COM_CD_D(models.Model):
    COM_DTL_CD = models.CharField(max_length=20, primary_key=True)
    COM_CD = models.ForeignKey('COM_CD_M', on_delete=models.CASCADE)
    COM_DTL_NM = models.CharField(max_length=20)
    REMARK_DC = models.CharField(max_length=20)
    USE_YN = models.CharField(max_length=20)

    def __str__(self):
        return self.COM_DTL_CD

class PHRASE(models.Model):
    PHRASE_SEQ = models.IntegerField(primary_key=True)
    PHRASE_KIND = models.CharField(max_length=20)
    QUOTE = models.CharField(max_length=300)
    PHRASE_FROM = models.CharField(max_length=100)
    PHRASE_URL = models.CharField(max_length=200)
    EMOTION_KIND = models.CharField(max_length=20)

    def __str__(self):
        return self.QUOTE

SEX_CHOICES = [
    ('M','Male'),
    ('F','Female'),
]

class USER(models.Model):
    EMAIL = models.CharField(max_length=100, primary_key=True)
    PASSWORD = models.CharField(max_length=50)
    USER_NM = models.CharField(max_length=20)
    SEX_CD = models.CharField(max_length=20, choices=SEX_CHOICES)
    USER_AGE = models.IntegerField()

    def __str__(self):
        return self.EMAIL

class FACE(models.Model):
    SMILE_SEQ = models.AutoField(primary_key=True)
    EMAIL = models.ForeignKey('USER', on_delete=models.CASCADE)
    STUDY_DATE = models.DateTimeField('date published')
    NEUTRAL_PATH = models.CharField(max_length=200)
    NEUTRAL_PERCENT = models.FloatField()
    SMILE1_PATH = models.CharField(max_length=200)
    SMILE1_PERCENT = models.FloatField()
    SMILE2_PATH = models.CharField(max_length=200)
    SMILE2_PERCENT = models.FloatField()
    SMILE3_PATH = models.CharField(max_length=200)
    SMILE3_PERCENT = models.FloatField()

    def __str__(self):
        return self.EMAIL


# Create your models here.

