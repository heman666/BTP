from django.db import models

class RainfallData(models.Model):
    city = models.CharField(max_length=100)
    mon_rainfall = models.FloatField('Monday')
    tue_rainfall = models.FloatField('Tuesday')
    wed_rainfall = models.FloatField('Wednesday')
    thu_rainfall = models.FloatField('Thursday')
    fri_rainfall = models.FloatField('Friday')
    sat_rainfall = models.FloatField('Saturday')
    sun_rainfall = models.FloatField('Sunday')
