FROM pytorch/pytorch:latest

# Arguments to set user and group IDs from the host
ARG USER_ID=1000
ARG GROUP_ID=1000

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

# Create a directory for Matplotlib config and set appropriate permissions
RUN mkdir -p /app/matplotlib_config && chown ${USER_ID}:${GROUP_ID} /app/matplotlib_config

# Set HOME and MPLCONFIGDIR so Matplotlib knows where to store config/cache
ENV HOME=/app
ENV MPLCONFIGDIR=/app/matplotlib_config

# Switch to non-root user (if desired), ensuring USER_ID and GROUP_ID are set by build args
RUN groupadd --gid ${GROUP_ID} appuser && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} -m appuser
USER appuser

# Default command
CMD ["python", "evaluate_models.py"]
