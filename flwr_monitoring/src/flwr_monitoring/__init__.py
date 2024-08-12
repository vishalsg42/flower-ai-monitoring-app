"""
This module provides functionality for monitoring and aggregating metrics in the flower-ai project.

Classes:
- GenericMonitoringStrategy: A class representing a generic monitoring strategy.
- Metrics: A module containing default metrics.
- MonitoringFactory: A module for creating monitoring tools.
- AggregateFunctions: A module containing functions for aggregating fit and evaluate metrics.

Attributes:
- __all__: A list of all the public objects in this module.

Usage:
Import the desired classes, modules, or attributes from this module to use them in your code.
"""
from .generic_monitoring_strategy import GenericMonitoringStrategy
from .metrics import default_metrics
from .monitoring_factory import create_monitoring_tool
from .aggregate_functions import aggregate_fit_metrics, aggregate_evaluate_metrics

__all__ = [
    "GenericMonitoringStrategy",
    "default_metrics",
    "create_monitoring_tool",
    "aggregate_fit_metrics",
    "aggregate_evaluate_metrics",
]
