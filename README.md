# furniture_classifier

## Intro

A deep neural network for classifying pictures of furniture as beds, chairs, or sofas. To get started, you can either build the Dockerfile or assemble manually.
The directory structure is as follows:

- `model.py`: The Pytorch model that acts as our classifier.
- `model.pk`: A selected best model from training on our dataset.
- `routes.py`: The API back-end.
- `training_loop.py`: Functions useful for training your own models.
- `data_preprocessing.py`: Functions and classes that were useful in augmenting and building our dataset.

If you are interested in the machine learning involved, start by reading `model.py`, and if you are interested in the API, `routes.py` is a good place for you to start.

## API Documentation

The API supports the following call(s):

### predict
- endpoint(s): `POST /predict`
- required data: image file
- returned status codes: 400 on error, 200 on success
- returned json:
```python
{"prediction": one of "bed", "chair", or "sofa"}
```
- example:
```python
import requests
with open("./bed.png", "rb") as payload:
    requests.post("http://localhost:5000/predict", files={'file': ("bed.png", payload)})
```

## Model Architecture

The model used to make predictions has the following architecture (note that all of the convolutional layers have a stride and padding of 2):

1. convolutional layer with:
    - 4 output channels
    - kernel size of 7
2. ReLU activation layer
3.  convolutional layer with:
    - 8 output channels
    - kernel size of 5
4. ReLU activation layer
5.  convolutional layer with:
    - 16 output channels
    - kernel size of 5
6. ReLU activation layer
7.  convolutional layer with:
    - 32 output channels
    - kernel size of 5
8. ReLU activation layer
9. fully connected layer with output size 128
10. dropout layer with 40% dropout
11. ReLU activation layer
12. fully connected layer with output size 3
