# QuantDesk on any always-on box (a home PC, a Raspberry Pi 5, an India-region VM).
# See docker-compose.yml: one container trades, one serves the phone app.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Kolkata
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
VOLUME /app/runtime
EXPOSE 8765
CMD ["python", "-m", "quantdesk", "serve", "--host", "0.0.0.0", "--port", "8765"]
