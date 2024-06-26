from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from strlit import main_loop


@csrf_exempt
def validate(request):
   if request.method == 'POST':
      aud___in = request.POST.get("aud___in")
      # if int(mus_flag) == 1: aud___in = f'♪ {aud___in} ♪'
      genre = request.POST.get("genre")
      length_penalty = request.POST.get("length_penalty", "1.")
      length_penalty = float(length_penalty)
      out___mp4_ = request.POST.get("output_location")
      main_loop(aud___in, genre, length_penalty=length_penalty, out___mp4_=out___mp4_)
      return HttpResponse('ok')