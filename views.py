from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from strlit import main_loop


@csrf_exempt
def validate(request):
   if request.method == 'POST':
      aud___in = request.POST.get("aud___in")
      genre = request.POST.get("genre")
      out___mp4_ = request.POST.get("output_location")
      max_length = request.POST.get("max_length")
      num_return_sequences = request.POST.get("num_return_sequences")
      num_beams = request.POST.get("num_beams")
      main_loop(
         aud___in,
         genre,
         out___mp4_=out___mp4_,
         max_length=int(max_length),
         num_return_sequences=int(num_return_sequences),
         num_beams=int(num_beams),
      )
      return HttpResponse('ok')