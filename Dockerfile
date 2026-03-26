FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY run.py .
COPY config.yaml .
COPY data.csv .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default command with arguments
# Docker will run with default paths inside container
CMD ["python", "run.py", "--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]
