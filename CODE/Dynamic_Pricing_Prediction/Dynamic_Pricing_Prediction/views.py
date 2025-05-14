from django.shortcuts import render
from users.forms import UserRegistrationForm

def Home(request):
    return render(request,'Home.html',{})
def Base(request):
    return render(request, 'base.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegister.html', {'form': form})