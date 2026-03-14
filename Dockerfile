FROM node:20

# Install Python + build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Node.js dependencies
RUN cd backend && npm install

# Setup Python virtual environment
RUN python3 -m venv /app/venv
RUN /app/venv/bin/pip install --upgrade pip

# Install requirements in virtualenv
RUN /app/venv/bin/pip install --no-cache-dir -r backend/requirements.txt

WORKDIR /app/backend

# Make sure to use virtualenv python if your backend runs Python
# If backend is Node only, keep npm start
CMD ["npm", "start"]