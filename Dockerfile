FROM python:3.11-slim

WORKDIR /app

# Install system utilities including libatomic1 required by Playwright driver node
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser and OS dependencies
RUN playwright install --with-deps chromium

# Copy codebase
COPY . .

# Expose port for Flask shopping store UI
EXPOSE 5000

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "store/app.py"]
