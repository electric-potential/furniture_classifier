FROM python:3.6.8
WORKDIR /app

COPY *.py .
COPY model.pk .
COPY requirements.txt .

RUN python -m pip install -r requirements.txt

CMD ["python","routes.py"]