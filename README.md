# CNN Author Classification: Kant vs Freud

A text classification project that uses a **Convolutional Neural Network (CNN)** to classify Spanish texts according to whether they were written by **Immanuel Kant** or **Sigmund Freud**.

**Live Demo:** [Try the web application →](https://cnnauthorclassserving-production.up.railway.app/)

The project covers the complete workflow from text preprocessing and model training to **SavedModel export, TensorFlow Serving, and a Django web application for inference**.

## Overview

The model was trained on **5,189 text files** containing approximately **2.45 million words** from Spanish translations of works by Kant and Freud.

The classification task itself was relatively straightforward. The main purpose of the project was to adapt and extend an implementation from Ganegedara's *Natural Language Processing with TensorFlow 2* to locally stored data, while also exploring the process of taking a trained NLP model through to a deployable inference service.

The original implementation was developed for TensorFlow 2. The code was adapted to work with newer TensorFlow/Keras versions and with this project's specific dataset and deployment requirements.

## Project Architecture

The project is divided into three main stages:

```text
                    TRAINING
                        │
                        ▼
              ┌──────────────────┐
              │  Text dataset    │
              │  Kant / Freud    │
              └────────┬─────────┘
                       │
                       ▼
              Text preprocessing
                       │
                       ▼
                  Tokenization
                       │
                       ▼
                CNN classification
                       │
                       ▼
              ┌──────────────────┐
              │  Trained model   │
              │  cnn_model.keras │
              └────────┬─────────┘
                       │
                       ▼
                 SavedModel export
                       │
                       ▼
              ┌──────────────────┐
              │ TensorFlow       │
              │ Serving          │
              └────────┬─────────┘
                       │
                 HTTP inference
                       │
                       ▼
              ┌──────────────────┐
              │ Django web app   │
              │                  │
              │ preprocessing    │
              │ + API request    │
              │ + result display │
              └──────────────────┘
```

The Django application does not load the Keras model directly. Instead, it preprocesses the input text and sends the resulting token sequence to a separate **TensorFlow Serving** instance through HTTP.

## Dataset

The dataset contains:

* **5,189 text files**
* Approximately **2,448,160 words**
* Two classes:

  * `Freud`
  * `Kant`

The original files are organized into two directories:

```text
data/
├── Freud/
└── Kant/
```

The dataset is split into:

* **80%** initial training set
* **20%** test set
* The training portion is subsequently split into:

  * **90% training**
  * **10% validation**

This results in:

| Split      | Samples |
| ---------- | ------: |
| Training   |   3,735 |
| Validation |     416 |
| Test       |   1,038 |

The test set contains 707 Freud texts and 331 Kant texts.

No exact duplicate texts were found between the training and test sets.

## Text Preprocessing

The preprocessing pipeline performs several operations before tokenization:

* Converts text to lowercase.
* Removes selected author names and publication-related terms.
* Removes phrases such as `sigmund freud`, `immanuel kant`, `obras completas` and editorial or sources names.
* Removes numbers and dates.
* Normalizes some characters and abbreviations.
* Keeps Spanish letters and whitespace.
* Removes excessive whitespace.
* Removes the first and last 10 words from texts longer than 20 words.

Removing author names is particularly important because otherwise the classifier could exploit explicit references to the author rather than relying on linguistic patterns present in the text.

The same preprocessing logic is reused during inference by the Django application.

## Tokenization

A Keras `Tokenizer` is fitted **only on the training data** to avoid incorporating information from the validation or test sets into the vocabulary.

The resulting vocabulary contains:

```text
52,113 tokens
```

The tokenized texts have a maximum selected sequence length of **599 tokens**, based on the 99th percentile of the training sequence-length distribution.

Sequences are:

* padded to 599 tokens,
* padded at the end (`post`),
* truncated at the end when necessary (`post`).

The tokenizer is exported to `tokenizer.json` so that the same vocabulary and tokenization process can be used during inference.

The preprocessing configuration is stored separately in:

```text
inference_config.json
```

This file contains the sequence length, label mapping, padding strategy, and truncation strategy.

## CNN Architecture

The classifier is implemented using TensorFlow/Keras.

The architecture consists of:

1. **Input layer**

   * Sequence length: 599

2. **Embedding layer**

   * Vocabulary size: 52,113
   * Embedding dimension: 64

3. **Three parallel 1D convolutional layers**

   * 100 filters each
   * Kernel sizes:

     * 3
     * 4
     * 5
   * ReLU activation

4. **Concatenation**

   * The three convolution outputs are concatenated.

5. **Global max pooling**

   * Reduces the sequence dimension while retaining the strongest detected features.

6. **Flatten layer**

7. **Output layer**

   * 2 neurons
   * Softmax activation
   * L2 regularization

The model contains approximately **3.41 million trainable parameters**.

```text
Input (599 tokens)
        │
        ▼
Embedding (64 dimensions)
        │
        ├──── Conv1D (kernel 3, 100 filters) ────┐
        │                                          │
        ├──── Conv1D (kernel 4, 100 filters) ────┤
        │                                          ├── Concatenate
        └──── Conv1D (kernel 5, 100 filters) ────┘
                                                   │
                                                   ▼
                                            Max Pooling
                                                   │
                                                   ▼
                                                Flatten
                                                   │
                                                   ▼
                                          Dense (2 classes)
                                                   │
                                                   ▼
                                               Softmax
```

## Training

The model was trained using:

* Optimizer: **Adam**
* Loss: `sparse_categorical_crossentropy`
* Batch size: **128**
* Epochs: **5**
* Learning-rate reduction: `ReduceLROnPlateau`

Training results:

| Epoch | Training Accuracy | Validation Accuracy |
| ----: | ----------------: | ------------------: |
|     1 |            69.69% |              76.44% |
|     2 |            89.88% |              96.63% |
|     3 |            98.31% |              98.32% |
|     4 |            99.57% |              99.04% |
|     5 |            99.79% |              99.28% |

The objective of the project was not extensive hyperparameter optimization. The five-epoch training configuration was sufficient for the experimental and deployment goals of the project.

## Test Results

On the held-out test set:

```text
Test accuracy: 99.42%
Test loss:     0.0338
```

Out of 1,038 test samples:

```text
Correctly classified: 1,032
Incorrectly classified: 6
```

The model therefore produced six classification errors on the test set.

The predictions and associated class probabilities were also exported to:

```text
df_clasificaciones.csv
df_clasificaciones.xlsx
```

## Model Export

After training, the Keras model is saved as:

```text
cnn_model.keras
```

For serving, the model is exported to TensorFlow's `SavedModel` format:

```text
kant_freud_model/
└── 1/
    ├── saved_model.pb
    ├── fingerprint.pb
    └── variables/
        ├── variables.data-00000-of-00001
        └── variables.index
```

The model version is currently `1`.

The exported model exposes the `serve` signature and expects:

```text
Input:
shape = (None, 599)
dtype = int32

Output:
shape = (None, 2)
dtype = float32
```

## SavedModel Validation

Before using the model with TensorFlow Serving, the exported `SavedModel` was loaded independently using TensorFlow and its predictions were compared with predictions produced directly by the original Keras model.

For the same inputs, both models produced matching predictions within the specified numerical tolerance.

This verifies that the exported model preserves the inference behavior of the trained Keras model.

## TensorFlow Serving

TensorFlow Serving is used as a separate inference service.

The Docker image is based on the official TensorFlow Serving image:

```dockerfile
FROM tensorflow/serving

COPY kant_freud_model /models/kant_freud_model

ENV MODEL_NAME=kant_freud_model
```

The model follows the TensorFlow Serving versioned-model directory convention:

```text
/models/kant_freud_model/1/
```

The Django application communicates with TensorFlow Serving through its HTTP prediction endpoint.

The model receives already-tokenized and padded sequences rather than raw text. This keeps the model-serving layer focused on model inference while preprocessing remains in the Django application.

## Django Application

The project includes a Django application that provides a simple interface for submitting text and obtaining a classification.

The application performs the following steps:

```text
User enters text
       │
       ▼
Django form validation
       │
       ▼
Text preprocessing
       │
       ▼
Tokenizer
       │
       ▼
Sequence padding
       │
       ▼
HTTP request to TensorFlow Serving
       │
       ▼
Softmax probabilities
       │
       ▼
Django interpretation
       │
       ▼
Result displayed to the user
```

### Input validation

The web form requires at least five words:

```text
Minimum input length: 5 words
```

### Prediction API

The Django application also exposes a prediction endpoint:

```text
POST /predict/
```

The request contains a text field:

```json
{
    "text": "..."
}
```

The application preprocesses the text and sends the resulting sequence to TensorFlow Serving.

The API returns the original text together with the predicted probabilities:

```json
{
    "text": "...",
    "freud": 0.995,
    "kant": 0.005
}
```

### Prediction interpretation

The web interface applies two probability thresholds:

```text
0.95 → high-confidence classification
0.85 → possible classification
< 0.85 → cannot be determined
```

This interpretation is performed by Django rather than by the neural network itself.

The underlying model always produces the two softmax probabilities. The thresholds are used only by the web interface to determine how the result is presented to the user.

## Example

A sample text can be passed through the same preprocessing and inference pipeline used by the application.

For example, a longer passage related to Freud produced:

```text
Freud: 0.9150
Kant:  0.0850

Prediction: Freud
```

The model can also produce uncertain results for texts that combine linguistic or conceptual characteristics associated with both authors.

## Project Structure

```text
.
├── CNN_Author_Classification_Kant_Freud_tfserving.ipynb
│
├── data/
│   ├── Freud/
│   └── Kant/
│
├── cnn_model.keras
├── tokenizer.json
├── inference_config.json
│
├── kant_freud_model/
│   └── 1/
│       ├── saved_model.pb
│       ├── fingerprint.pb
│       └── variables/
│
├── Dockerfile.tfserving
│
├── df_clasificaciones.csv
├── df_clasificaciones.xlsx
│
├── manage.py
│
├── classifier/
│   ├── forms.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── views.py
│   ├── tests.py
│   └── templates/
│       └── predict.html
│
└── cnn_web_kf/
    ├── settings.py
    ├── urls.py
    ├── asgi.py
    └── wsgi.py
```

Dataset: The original text corpus is not included in the repository due to copyright and redistribution considerations. The data/Freud/ and data/Kant/ directories represent the expected dataset structure.

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* scikit-learn
* Django
* TensorFlow Serving
* Docker

## Reference

The CNN architecture was adapted from the examples presented by Ganegedara in:

*Natural Language Processing with TensorFlow 2*

The original implementation can be found in the author's accompanying repository:

`thushv89/packt_nlp_tensorflow_2`

## Main Learning Objectives

This project was developed to explore the complete lifecycle of an NLP classification model:

* Loading and preparing a locally stored text corpus.
* Building a reproducible preprocessing pipeline.
* Training a CNN for text classification.
* Exporting a trained Keras model to `SavedModel`.
* Validating the exported model independently.
* Serving the model through TensorFlow Serving.
* Building a Django application around the inference service.
* Keeping training-time and inference-time preprocessing consistent.
* Returning model probabilities through an HTTP API.
* Applying application-level confidence thresholds to model predictions.
