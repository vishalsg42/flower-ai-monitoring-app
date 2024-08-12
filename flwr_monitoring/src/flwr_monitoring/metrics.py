from prometheus_client import Summary, Counter, Gauge, Histogram
"""
Metrics module for monitoring Flower AI.

This module defines a list of default metrics used for monitoring the performance of Flower AI. Each metric is represented as a dictionary with the following keys:
- name: The name of the metric.
- description: A brief description of the metric.
- type: The type of the metric (Summary, Counter, Gauge, or Histogram).

The default_metrics list contains the following metrics:
- flower_accuracy: Average accuracy per round.
- flower_loss: Average loss per round.
- flower_client_participation: Number of participating clients per round.
- flower_training_time_seconds: Time taken for training per round.
- flower_communication_time_seconds: Time taken for communication per round.
- cpu_usage: CPU usage.
- memory_usage: Memory usage.
- gpu_usage: GPU usage.
- clients_selected: Number of clients selected per round.
- current_epoch: Current epoch number.
"""

default_metrics = [
    {"name": "flower_accuracy", "description": "Average accuracy per round", "type": Summary},
    {"name": "flower_loss", "description": "Average loss per round", "type": Summary},
    {"name": "flower_client_participation", "description": "Number of participating clients per round", "type": Gauge},
    {"name": "flower_training_time_seconds", "description": "Time taken for training per round", "type": Histogram},
    {"name": "flower_communication_time_seconds", "description": "Time taken for communication per round", "type": Histogram},
    {"name": "cpu_usage", "description": "CPU usage", "type": Gauge},
    {"name": "memory_usage", "description": "Memory usage", "type": Gauge},
    {"name": "gpu_usage", "description": "GPU usage", "type": Gauge},
    {"name": "clients_selected", "description": "Number of clients selected per round", "type": Gauge},
    {"name": "current_epoch", "description": "Current epoch number", "type": Gauge},
]
