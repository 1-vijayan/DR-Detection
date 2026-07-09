# Use python 3.9 slim base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and general operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (default for HF Spaces is 7860)
EXPOSE 7860

# Run using gunicorn, dynamically binding to $PORT (Render) or defaulting to 7860 (Hugging Face)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} app:app"]
