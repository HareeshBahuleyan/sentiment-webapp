# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2016-11-09 03:18
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.CharField(max_length=10000)),
                ('title', models.CharField(max_length=1000)),
                ('comment', models.CharField(max_length=1000)),
                ('userid', models.CharField(max_length=8)),
                ('articleid', models.CharField(max_length=10)),
                ('version', models.CharField(max_length=10)),
                ('date', models.DateTimeField()),
                ('status', models.IntegerField()),
                ('html_output', models.CharField(max_length=10000)),
            ],
        ),
    ]