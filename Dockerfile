FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HUMEO_TRANSCRIBE_PROVIDER=openai \
    PORT=7860

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && \
    pip install ./humeo-core && \
    pip install . gradio

EXPOSE 7860

CMD ["python", "app.py"]
