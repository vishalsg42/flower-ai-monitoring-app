#!/bin/bash

# Function to fetch the EC2 instance's IP address
fetch_ec2_ip() {
  curl -s http://169.254.169.254/latest/meta-data/local-ipv4
}

# Parse arguments
while getopts ":i:p:m:" opt; do
  case ${opt} in
    i )
      SERVER_IP=$OPTARG
      ;;
    p )
      SERVER_PORT=$OPTARG
      ;;
    m )
      PROMETHEUS_IP=$OPTARG
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

# Default port if not provided
SERVER_PORT=${SERVER_PORT:-8080}

# Fetch EC2 IP if not provided
if [ -z "$SERVER_IP" ]; then
  SERVER_IP=$(fetch_ec2_ip)
  if [ -z "$SERVER_IP" ]; then
    echo "Error: Could not determine the server IP address." 1>&2
    exit 1
  fi
fi

# Default Prometheus IP if not provided
PROMETHEUS_IP=${PROMETHEUS_IP:-localhost}

echo "Starting server with IP: $SERVER_IP on port: $SERVER_PORT and Prometheus IP: $PROMETHEUS_IP"

# Run the Docker container for the server
docker run  -e SERVER_IP="$SERVER_IP" -e SERVER_PORT="$SERVER_PORT" -e PROMETHEUS_IP="$PROMETHEUS_IP" -p "$SERVER_PORT:$SERVER_PORT" my_server_image python server.py --ip "$SERVER_IP" --port "$SERVER_PORT" --prometheus-ip "$PROMETHEUS_IP"

