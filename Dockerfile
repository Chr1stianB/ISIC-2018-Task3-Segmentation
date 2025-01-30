FROM pytorch/pytorch:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    pip install --no-cache-dir pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg \
       python-gdcm pydicom

# Copy the rest of the application code
COPY . /app

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Default command
CMD ["python", "main.py"]
