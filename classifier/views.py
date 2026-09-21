import requests

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .forms import PredictionForm
from .preprocessing import preprocess_text


import os

# From railway variables
TF_SERVING_URL = os.environ["TF_SERVING_URL"]


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Use POST."},
            status=405
        )

    text = request.POST.get("text")

    if not text:
        return JsonResponse(
            {"error": "Missing 'text'."},
            status=400
        )

    sequence = preprocess_text(text)

    payload = {
        "signature_name": "serve",
        "instances": sequence.tolist()
    }

    response = requests.post(
        TF_SERVING_URL,
        json=payload
    )

    response.raise_for_status()

    predictions = response.json()["predictions"][0]

    return JsonResponse({
        "text": text,
        "freud": predictions[0],
        "kant": predictions[1]
    })


THRESHOLD_DUDOSO = 0.85
THRESHOLD_SEGURO = 0.95


def predict_page(request):

    form = PredictionForm(request.POST or None)
    result = None

    if request.method == "POST" and form.is_valid():

        text = form.cleaned_data["text"]

        sequence = preprocess_text(text)

        payload = {
            "signature_name": "serve",
            "instances": sequence.tolist()
        }

        response = requests.post(
            TF_SERVING_URL,
            json=payload
        )

        response.raise_for_status()

        predictions = response.json()["predictions"][0]

        freud = predictions[0]
        kant = predictions[1]

        if freud >= THRESHOLD_SEGURO:
            label = "Muy probablemente sería algo que diria Freud"
            show_probabilities = True

        elif kant >= THRESHOLD_SEGURO:
            label = "Muy probablemente sería algo que diria Kant"
            show_probabilities = True

        elif freud >= THRESHOLD_DUDOSO:
            label = "Tal vez sea algo que diria Freud"
            show_probabilities = False

        elif kant >= THRESHOLD_DUDOSO:
            label = "Tal vez sea algo que diria Kant"
            show_probabilities = False

        else:
            label = "No puede determinarse"
            show_probabilities = False

        result = {
            "label": label
        }

        if show_probabilities:
            result["freud"] = freud
            result["kant"] = kant

    return render(
        request,
        "predict.html",
        {
            "form": form,
            "result": result
        }
    )