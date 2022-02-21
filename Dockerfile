FROM python:3.6-buster

ENV HOME /root
WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]