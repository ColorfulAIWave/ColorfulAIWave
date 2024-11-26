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
RUN pip3 install python-multipart pyarmor

# Set the working directory in the container
WORKDIR /app

# Copy all files, including pytransform
COPY . .

# Ensure pytransform directory exists before running pyarmor download
RUN mkdir -p /app/pytransform/platforms && \
    arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        platform="linux.x86_64"; \
    elif [ "$arch" = "aarch64" ]; then \
        platform="linux.aarch64"; \
    else \
        echo "Unsupported architecture: $arch" && exit 1; \
    fi && \
    pyarmor download --platform $platform && \
    ls -la /app/pytransform/platforms

# Verify the parent pytransform directory for troubleshooting
RUN ls -la /app/pytransform/

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port for FastAPI
EXPOSE 8000

# Command to run FastAPI on uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
