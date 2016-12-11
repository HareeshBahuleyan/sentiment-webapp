from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import Http404
from django.shortcuts import render, get_object_or_404

from mlclassifiers import all_taggers

# Create your views here.

def index(request):

	template = loader.get_template('documents/detail.html')
	context = {
		'descript':"Enter text below, select classifier and threshold; then click Highlight:",
	}

	if(request.GET.get('highlighter')):
		classifier = request.GET.get('classifier')
		threshold = request.GET.get('threshold')
		text = request.GET.get('text')
		highlight_text = all_taggers.highlighter(request.GET.get('text'), classifier, threshold)
		return render(request, 'documents/highlightdetail.html', {'htext':highlight_text, 'text':text, 'classifier':classifier + " @ Threshold - " + threshold})


	return HttpResponse(template.render(context, request))

