# to make sure the gpu is detected, run docker with the following:
# --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888
# optional : -it for interactive terminal
#            --rm : remove container after run
#            -v "$(pwd)/data:/app/data" : mount data dir

FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IN_DOCKER=true

# Define repo generated in the container
WORKDIR /app

COPY . .

RUN mkdir -p data/raw data/processed models logs

EXPOSE 5000
EXPOSE 8888

CMD ["bash", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/app"]
