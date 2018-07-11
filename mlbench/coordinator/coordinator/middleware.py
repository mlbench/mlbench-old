from constance import config
from django.http import HttpResponseRedirect
from django.urls import reverse


class FirstVisitMiddleware(object):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if (config.FIRST_TIME
                and not request.path.startswith("/first_time")):
            return HttpResponseRedirect(reverse("first_time"))
        return self.get_response(request)
