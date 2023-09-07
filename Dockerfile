# A Dockerfile is a text document that contains all the commands
# a user could call on the command line to assemble an image.

FROM python:3.9.18-slim-bullseye

# Our Debian with python is now installed.
# Imagine we have folders /sys, /tmp, /bin etc. there
# like we would install this system on our laptop.

RUN mkdir build

# We create folder named build for our stuff.

WORKDIR /build

# Basic WORKDIR is just /
# Now we just want to our WORKDIR to be /build

COPY . .

# FROM [path to files from the folder we run docker run]
# TO [current WORKDIR]
# We copy our files (files from .dockerignore are ignored)
# to the WORKDIR

RUN apt update

RUN apt install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8444

WORKDIR /build

CMD python -m uvicorn app.main:app --host 0.0.0.0 --port 8444
