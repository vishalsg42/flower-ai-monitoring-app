from abc import ABC, abstractmethod

class BaseMonitoringStrategy(ABC):
    """
    Abstract base class for monitoring strategies.

    Attributes:
        metrics (list): List of metrics to be monitored.
        config (dict): Configuration parameters for the monitoring strategy.

    Methods:
        initialize_metrics(): Initializes the metrics for monitoring.
        track_resource_usage(): Tracks the resource usage.
        observe_metric(metric_name, value): Observes a specific metric.
        start_server(): Starts the monitoring server.
    """
    def __init__(self, metrics, config):
        self.metrics = metrics
        self.config = config

    @abstractmethod
    def initialize_metrics(self):
        """
        Abstract method for initializing metrics in a monitoring strategy.

        This method should be implemented by subclasses to initialize the metrics used in the monitoring strategy.

        Parameters:
            None

        Returns:
            None
        """
        pass

    @abstractmethod
    def track_resource_usage(self):
        """
        Abstract method for tracking resource usage.

        This method should be implemented by subclasses to track the resource usage of a monitoring strategy.
        """
        pass

    @abstractmethod
    def observe_metric(self, metric_name, value):
        """
        Abstract method for observing a metric.

        Parameters:
        - metric_name (str): The name of the metric.
        - value: The value of the metric.

        Returns:
        None
        """
        pass

    @abstractmethod
    def start_server(self):
        """
        Starts the server for monitoring the base strategy.
        """
        pass
