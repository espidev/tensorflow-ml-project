FROM python:3.6.9-buster
WORKDIR /usr/src/tensorflow-ml-project
RUN apt-get update && pip install opencv-python \
    tqdm \
    numpy \
    tensorflow \
    keras \
    matplotlib \
    pandas \
    seaborn \
    gdown

RUN apt-get install -y vim
COPY . .
CMD ["python", "pipeline.py"]
