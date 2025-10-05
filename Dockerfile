# Stock Forecasting Application - Multi-stage Docker Build

# Stage 1: Backend
FROM python:3.9-slim as backend

WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ .

EXPOSE 5000

# Stage 2: Frontend
FROM node:18-alpine as frontend

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend source code
COPY frontend/ .

# Build the React app
RUN npm run build

EXPOSE 3000

# Stage 3: Final image with both backend and frontend
FROM python:3.9-slim

# Install Node.js for serving frontend
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend from stage 1
COPY --from=backend /app/backend ./backend
COPY --from=backend /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin

# Copy frontend build from stage 2
COPY --from=frontend /app/frontend/build ./frontend/build
COPY --from=frontend /app/frontend/package*.json ./frontend/

# Install serve to run the React build
RUN npm install -g serve

# Copy startup script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Expose ports
EXPOSE 3000 5000

# Start both services
ENTRYPOINT ["./docker-entrypoint.sh"]