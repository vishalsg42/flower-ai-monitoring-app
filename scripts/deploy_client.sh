#!/bin/bash

# Function to fetch the EC2 instance's IP address
fetch_ec2_ip() {
  curl -s http://169.254.169.254/latest/meta-data/local-ipv4
}

# Parse arguments
while getopts ":i:p:c:u:" opt; do
  case ${opt} in
    i )
      SERVER_IP=$OPTARG
      ;;
    p )
      SERVER_PORT=$OPTARG
      ;;
    c )
      CLIENT_ID=$OPTARG
      ;;
    u )
      USE_CUDA=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done

# Default values if not provided
SERVER_PORT=${SERVER_PORT:-8080}
CLIENT_ID=${CLIENT_ID:-0}
USE_CUDA=${USE_CUDA:-false}

# Fetch EC2 IP if not provided
if [ -z "$SERVER_IP" ]; then
  SERVER_IP=$(fetch_ec2_ip)
  if [ -z "$SERVER_IP" ]; then
    echo "Error: Could not determine the server IP address." 1>&2
    exit 1
  fi
fi

echo "Starting client with server IP: $SERVER_IP on port: $SERVER_PORT with client ID: $CLIENT_ID"

# Run the Docker container for the client
docker run -d -e SERVER_IP="$SERVER_IP" -e SERVER_PORT="$SERVER_PORT" -e CLIENT_ID="$CLIENT_ID" -e USE_CUDA="$USE_CUDA" my_client_image python client.py --server-ip "$SERVER_IP" --server-port "$SERVER_PORT" --client-id "$CLIENT_ID" --use-cuda "$USE_CUDA"
