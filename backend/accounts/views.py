from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import CreateView


class SignUpView(CreateView):
    form_class = UserCreationForm          # Django 기본 회원가입 폼
    template_name = "accounts/signup.html"
    success_url = reverse_lazy("login")    # 가입 성공 → 로그인 화면


@login_required                            # 로그인 안 했으면 로그인 화면으로 보냄
def home(request):
    return render(request, "home.html")