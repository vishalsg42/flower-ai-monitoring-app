#!/bin/bash

# Function to fetch the EC2 instance's IP address
fetch_ec2_ip() {
  TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
  curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/local-ipv4
}

# Default values
# Initialize variables with default values
SERVER_IP=""
SERVER_PORT=8080
PROMETHEUS_IP="localhost"
PROMETHEUS_PORT=9090
IMAGE_NAME="flower-ai-monitoring-app"
DOCKERFILE_PATH="."

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --server-ip)
      SERVER_IP="$2"
      shift # past argument
      shift # past value
      ;;
    --server-port)
      SERVER_PORT="$2"
      shift # past argument
      shift # past value
      ;;
    --prometheus-ip)
      PROMETHEUS_IP="$2"
      shift # past argument
      shift # past value
      ;;
    --prometheus-port)
      PROMETHEUS_PORT="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Fetch EC2 IP if server-ip not provided
if [ -z "$SERVER_IP" ]; then
  SERVER_IP=$(fetch_ec2_ip)
  if [ -z "$SERVER_IP" ]; then
    echo "Error: Could not determine the server IP address." 1>&2
    exit 1
  fi
fi

# Default Prometheus IP if not provided
PROMETHEUS_IP=${PROMETHEUS_IP:-localhost}


echo "Building Docker image $IMAGE_NAME from $DOCKERFILE_PATH"

# Build the Docker image
docker build -t "$IMAGE_NAME" "$DOCKERFILE_PATH"

if [ $? -ne 0 ]; then
  echo "Error: Failed to build the Docker image." 1>&2
  exit 1
fi

echo "Starting server with IP: $SERVER_IP on port: $SERVER_PORT, Prometheus IP: $PROMETHEUS_IP  and Prometheus port: $PROMETHEUS_PORT"

# Run the Docker container for the server
docker run -e SERVER_IP="$SERVER_IP" -e SERVER_PORT="$SERVER_PORT" -e PROMETHEUS_IP="$PROMETHEUS_IP" -e PROMETHEUS_PORT="$PROMETHEUS_PORT" -p "$SERVER_PORT:$SERVER_PORT" flower-ai-monitoring-app --server-ip "$SERVER_IP" --server-port "$SERVER_PORT" --prometheus-ip "$PROMETHEUS_IP" --prometheus-port "$PROMETHEUS_PORT"
