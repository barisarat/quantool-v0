from django.shortcuts import render
from . import btc_model # get the model codes

def home(request):
    return render(request, 'index.html')

def result(request):
    user_input = int(request.GET['user_input'])
    user_input = btc_model.btc_predict(user_input)
    return render(request, 'result.html', {'home_input':user_input})
