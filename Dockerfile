FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FEDSECUREAI_PRELOAD_MODELS=true \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^torch(==|>=|<=|$)' requirements.txt > /tmp/req_nocuda.txt && \
    pip install --no-cache-dir -r /tmp/req_nocuda.txt && \
    rm -f /tmp/req_nocuda.txt

COPY asgi.py .
COPY backend ./backend

EXPOSE 7860

CMD ["sh", "-c", "uvicorn asgi:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
