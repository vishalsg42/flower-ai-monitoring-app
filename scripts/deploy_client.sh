#!/bin/bash

# Function to fetch the EC2 instance's IP address
fetch_ec2_ip() {
  TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
  curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/local-ipv4
}

# Default values
SERVER_IP=""
SERVER_PORT=8080
CLIENT_ID=0
USE_CUDA=false
IMAGE_NAME="flower-ai-client-app"
DOCKERFILE_PATH="DockerfileClient"  # Set the default Dockerfile path to DockerfileClient

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
    --client-id)
      CLIENT_ID="$2"
      shift # past argument
      shift # past value
      ;;
    --use-cuda)
      USE_CUDA="$2"
      shift # past argument
      shift # past value
      ;;
    --image-name)
      IMAGE_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    --dockerfile-path)
      DOCKERFILE_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Fetch EC2 IP if not provided
if [ -z "$SERVER_IP" ]; then
  SERVER_IP=$(fetch_ec2_ip)
  if [ -z "$SERVER_IP" ]; then
    echo "Error: Could not determine the server IP address." 1>&2
    exit 1
  fi
fi

echo "Building Docker image $IMAGE_NAME using Dockerfile from $DOCKERFILE_PATH"

# Build the Docker image with the specified Dockerfile
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .

if [ $? -ne 0 ]; then
  echo "Error: Failed to build the Docker image." 1>&2
  exit 1
fi

echo "Starting client with server IP: $SERVER_IP on port: $SERVER_PORT with client ID: $CLIENT_ID and use_cuda: $USE_CUDA"

# Run the Docker container for the client
docker run -e SERVER_IP="$SERVER_IP" -e SERVER_PORT="$SERVER_PORT" -e CLIENT_ID="$CLIENT_ID" -e USE_CUDA="$USE_CUDA" "$IMAGE_NAME" --server-ip "$SERVER_IP" --server-port "$SERVER_PORT" --client-id "$CLIENT_ID" --use-cuda "$USE_CUDA"
