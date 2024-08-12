from abc import ABC, abstractmethod


class BaseMonitoringTool(ABC):
    def __init__(self, config, metrics):
        """
        Initializes the BaseMonitoringTool object.

        Args:
            config (dict): The configuration for the monitoring tool.
            metrics (list): The list of metrics to be monitored.

        Returns:
            None
        """
        self.config = config
        self.metrics = metrics

    @abstractmethod
    def initialize_metrics(self):
        """
        Initializes the metrics for the monitoring tool.
        """
        pass

    @abstractmethod
    def start_monitoring(self):
        """
        Starts the monitoring process.
        """
        pass

    @abstractmethod
    def track_resource_usage(self):
        """
        Tracks the resource usage of the monitoring tool.

        This method is responsible for monitoring and tracking the resource usage of the monitoring tool.
        It can be used to collect data on CPU usage, memory consumption, disk usage, network activity, etc.

        Parameters:
            self (BaseMonitoringTool): The instance of the monitoring tool.

        Returns:
            None
        """
        pass

    @abstractmethod
    def observe_metric(self, metric_name, value):
        """
        Observes a metric and its corresponding value.

        Parameters:
        - metric_name (str): The name of the metric.
        - value: The value of the metric.

        Returns:
        - None
        """
        pass
