# Generated by Django 2.0.7 on 2018-08-14 11:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_auto_20180803_1550'),
    ]

    operations = [
        migrations.AddField(
            model_name='kubemetric',
            name='cumulative',
            field=models.BooleanField(default=False),
        ),
    ]
