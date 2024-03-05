# YOLO Document Layout Analysis Server

This server provides endpoints to detect:
* Blocks (text blocks and images)
* Text lines
* Words
* Glyphs

Each detection is independent - it's up to the client software to combine them.

Only block detection provides two different classes, "TopLevelTextBlock" and "Illustration".

Behind-the-scenes, all models were constructed using [YOLOv8](https://docs.ultralytics.com/).

## Running locally

If you haven't done so, create a pyenv environment.

Install [pyenv](https://brain2life.hashnode.dev/how-to-install-pyenv-python-version-manager-on-ubuntu-2004)

```shell
pyenv update
pyenv install 3.9.18
pyenv global 3.9.18
pyenv virtualenv yolo-dla-server
pyenv activate yolo-dla-server
```

To install all Python requirements:
```shell
pip install -r requirements.txt 
```

To run locally:

```shell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8444
```

Then navigate to:
```shell
http://localhost:8444/docs
```

# Running as a docker image

```shell
docker build -t registry.gitlab.com/jochre/yolo-dla-server:[YOUR-TAG] .
docker run --rm -it  -p 8444:8444/tcp registry.gitlab.com/jochre/yolo-dla-server:[YOUR_TAG]
```

Then navigate to:
```shell
http://localhost:8444/docs
```

# Working with the remote docker repository

## Logging in

```shell
docker login registry.gitlab.com
```

## Pulling an image and running it locally

After login, to download the docker image from the repository and run it locally:
```shell
docker pull registry.gitlab.com/jochre/yolo-dla-server:[YOUR_TAG]
docker run --rm -it -p 8444:8444/tcp registry.gitlab.com/jochre/yolo-dla-server:[YOUR_TAG]
```

Then navigate to:
```shell
http://localhost:8444/docs
```

# Pushing a docker image

After login, to build the image and push it, run the commands below.
If you don't enter a tag (below `[YOUR_TAG]`, the tag `latest` is added automatically).

```shell
docker build -t registry.gitlab.com/jochre/yolo-dla-server:[YOUR-TAG] .
docker push registry.gitlab.com/jochre/yolo-dla-server:[YOUR-TAG]
```