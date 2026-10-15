FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CREDIT_RISK_REQUIRE_ARTIFACT=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json, os, urllib.request; port=os.environ.get('PORT', '8000'); status=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=3)); assert status.get('status') == 'ok' and status.get('model_mode') == 'production']

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
