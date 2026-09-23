# Use official Python 3.10 slim Linux image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr and creating .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install essential Linux system dependencies for OpenCV and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (standard requirement for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set up working directory
WORKDIR /app

# Ensure proper user permissions for app and home cache directories
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and model weights
COPY --chown=user:user . .

# Ensure runtime directories exist for uploads, evidence, and model caches
RUN mkdir -p /app/integrated/uploads \
             /app/integrated/evidence \
             /home/user/.paddlex \
             /home/user/.paddleocr \
             /home/user/.config/Ultralytics

# Working directory set to integrated where app.py and templates reside
WORKDIR /app/integrated

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run Flask built-in server (required so the background AI thread starts)
CMD ["python", "app.py"]
