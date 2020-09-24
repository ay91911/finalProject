from django.db import models

class Phrase(models.Model):
    Num = models.IntegerField(primary_key=True)
    Category = models.CharField(max_length=20)
    Quote = models.CharField(max_length=300)
    Person = models.CharField(max_length=100)
    URL = models.CharField(max_length=200)
    Emotion = models.CharField(max_length=20)

    def __str__(self):
        return self.Quote

class User(models.Model):
    Email = models.CharField(max_length=100, primary_key=True)
    Password = models.CharField(max_length=50)
    UserName = models.CharField(max_length=20)
    Gender = models.CharField(max_length=20)
    Age = models.IntegerField()

    def __str__(self):
        return self.Email

class Face(models.Model):
    Num = models.IntegerField(primary_key=True)
    Email = models.ForeignKey(User, on_delete=models.CASCADE)
    Date = models.DateTimeField('date published')
    Neutral_path = models.CharField(max_length=200)
    Neutral_accuracy = models.FloatField()
    Smile1_path = models.CharField(max_length=200)
    Smile1_accuracy = models.FloatField()
    Smile2_path = models.CharField(max_length=200)
    Smile2_accuracy = models.FloatField()
    Smile3_path = models.CharField(max_length=200)
    Smile3_accuracy = models.FloatField()

    def __str__(self):
        return self.Email


# Create your models here.

