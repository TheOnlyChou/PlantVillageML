FROM python:3.10

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IN_DOCKER=true

# Define repo generated in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN mkdir -p data/raw data/processed models logs

EXPOSE 5000

CMD ["/bin/bash"]