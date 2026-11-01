FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG UID=1000
ARG GID=1000

RUN groupadd --gid ${GID} appuser \
    && useradd --uid ${UID} --gid ${GID} --create-home appuser

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]