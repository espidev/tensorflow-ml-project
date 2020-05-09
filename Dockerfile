FROM tensorflow/tensorflow:latest-py3
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

COPY . .



