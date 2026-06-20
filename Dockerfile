FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ /app/src/
COPY dashboard/ /app/dashboard/
COPY 유홍성/ /app/유홍성/
COPY start.sh /app/start.sh

# Grant execution rights on entrypoint script
RUN chmod +x /app/start.sh

# Environment variables
ENV PYTHONPATH=/app/src
ENV SNCT_BASE_DIR=/app

# Expose ports
EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["/app/start.sh"]
