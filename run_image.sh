docker run -it --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8888:8888 -p 5000:5000 \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  plantvillageml-container
