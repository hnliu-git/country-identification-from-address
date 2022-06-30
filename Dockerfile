FROM python:3.8

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

#CMD
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
