# Use a lightweight Ubuntu base image
FROM ubuntu:20.04

# Prevent interactive prompts during package installations
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install PyArmor (specific version 7.x)
RUN pip3 install pyarmor==7.6.0

# Set the working directory in the container
WORKDIR /app

# Copy pre-downloaded pytransform runtime files
COPY ./pytransform /app/pytransform
RUN ls -la /app/pytransform

# Copy application files
COPY . .

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port for FastAPI
EXPOSE 8000

# Command to run FastAPI on uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
