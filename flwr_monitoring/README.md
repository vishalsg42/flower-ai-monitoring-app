# Federated Learning Monitoring Library

## Overview

The Federated Learning Monitoring Library is designed to provide comprehensive monitoring capabilities for federated learning processes. This library extends existing federated learning strategies (like FedAvg) with monitoring tools such as Prometheus. It allows users to track various metrics related to training, communication, and resource usage, providing deep insights into the performance and efficiency of federated learning systems.

## Features

- **Custom Monitoring Strategy**: Wraps existing federated learning strategies with monitoring capabilities.
- **Prometheus Integration**: Supports Prometheus as a monitoring tool out-of-the-box.
- **Resource Usage Tracking**: Monitors CPU, memory, and GPU usage.
- **Comprehensive Metrics**: Tracks training time, communication time, client participation, accuracy, loss, and more.

## Installation

To install the library, clone the repository and install the dependencies using pip:

```bash
git clone git@github.com:kandola-network/KanFL.git
cd KanFL
pip install -r requirements.txt

