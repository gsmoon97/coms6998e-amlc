FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY app.py .

RUN pip install --no-cache-dir --break-system-packages torchvision flask pillow

CMD ["python", "app.py"]
