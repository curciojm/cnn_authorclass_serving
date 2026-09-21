from django import forms


class PredictionForm(forms.Form):

    text = forms.CharField(
        label="Texto",
        widget=forms.Textarea(
            attrs={
                "rows": 10,
                "placeholder": "Escribí un texto..."
            }
        )
    )

    def clean_text(self):
        text = self.cleaned_data["text"]

        word_count = len(text.split())

        if word_count < 5:
            raise forms.ValidationError(
                "El texto debe contener al menos 5 palabras."
            )

        return text