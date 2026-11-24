import os

from celery import Celery

# Celery가 Django 설정을 쓰도록 지정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")

# settings.py에서 CELERY_로 시작하는 설정을 읽음
app.config_from_object("django.conf:settings", namespace="CELERY")

# 등록된 앱들의 tasks.py를 자동으로 찾음
app.autodiscover_tasks()
