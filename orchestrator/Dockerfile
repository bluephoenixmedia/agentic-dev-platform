# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Install curl for network debugging
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the dependencies file from the project root
COPY requirements.txt .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the local project directory into the container
COPY . .

# 7. Command to keep the container running
CMD ["tail", "-f", "/dev/null"]

