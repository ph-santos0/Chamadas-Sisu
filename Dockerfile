# Dockerfile para a V5.1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libfreetype6-dev libjpeg62-turbo-dev zlib1g-dev curl  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY afirmativas_app_v5_1.py /app/afirmativas_app_v5_1.py

EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "afirmativas_app_v5_1.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
