# --- Stage 1: Build Stage ---
# Use a full Python image to handle complex builds.
FROM python:3.13 AS build

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Stage ---
# Use a much smaller, runtime-only Python image.
FROM python:3.13-slim

WORKDIR /app

# Copy only the installed packages from the build stage
COPY --from=build /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Copy your application code
COPY . .

# Set the port and command
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]