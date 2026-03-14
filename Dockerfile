# Base image with Node
FROM node:20

# Install Python + build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the whole project
COPY . .

# Node dependencies
RUN cd backend && npm install

# Python virtual environment for embedding service
RUN python3 -m venv /app/venv
RUN /app/venv/bin/pip install --upgrade pip

# Install Python dependencies inside venv
RUN /app/venv/bin/pip install --no-cache-dir -r backend/requirements.txt

# Expose port Railway expects
ENV PORT 3000
EXPOSE 3000

# Set working directory for Node service
WORKDIR /app/backend

# CMD: ensure Node can call Python via venv
# Node can now do: /app/venv/bin/python embedder.py
CMD ["sh", "-c", "echo Starting Node backend... && npm start"]