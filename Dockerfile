FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY app.py .
COPY model/ ./model/

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]