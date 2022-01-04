FROM python:3.8-slim

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ADD . /app
RUN pip install /app

ENTRYPOINT ["RandomForest"]
CMD ["--help"]