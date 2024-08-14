# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the flwr_monitoring package and install it
COPY flwr_monitoring /app/flwr_monitoring
RUN pip install /app/flwr_monitoring

# Copy the rest of the application code
COPY . /app

# Define the command to run the server
# CMD ["python", "server.py"]
ENTRYPOINT ["python", "server.py"]
