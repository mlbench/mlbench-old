from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
    
    return HttpResponse("Welcome! yaynayfsdfsdfxcvxcvxd!")

def first_time(request):
    return HttpResponse("First Time!")