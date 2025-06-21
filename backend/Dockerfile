FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt ./

# Install dependencies globally (no virtual environment)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a startup script that uses the renamed application variable
RUN echo '#!/bin/bash\nexec python -m uvicorn app.main:application --host 0.0.0.0 --port 8000' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Use the startup script with renamed application variable
CMD ["/app/start.sh"]
