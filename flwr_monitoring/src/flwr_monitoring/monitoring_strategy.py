import flwr as fl
from flwr.common import Parameters, EvaluateRes, FitRes, Metrics
from typing import List, Tuple
import time
from .prometheus_monitoring import PrometheusMonitoring
from .metrics import default_metrics

class MonitoringStrategy(fl.server.strategy.Strategy):
    def __init__(self, base_strategy, monitoring_tool="prometheus", metrics=default_metrics, config=None):
        """
        Initialize the MonitoringStrategy object.

        Args:
            base_strategy (object): The base strategy object.
            monitoring_tool (str, optional): The monitoring tool to be used. Defaults to "prometheus".
            metrics (list, optional): The list of metrics to be monitored. Defaults to default_metrics.
            config (dict, optional): The configuration for the monitoring tool. Defaults to None.
        """
        self.base_strategy = base_strategy
        self.monitoring = self._initialize_monitoring_tool(monitoring_tool, metrics, config)

    def _initialize_monitoring_tool(self, tool, metrics, config):
        """
        Initializes the monitoring tool based on the specified tool name.

        Args:
            tool (str): The name of the monitoring tool.
            metrics (list): The list of metrics to be monitored.
            config (dict): The configuration for the monitoring tool.

        Returns:
            MonitoringTool: An instance of the monitoring tool.

        Raises:
            ValueError: If the specified monitoring tool is not supported.
        """
        if tool == "prometheus":
            port = config.get("port", 8000)
            return PrometheusMonitoring(metrics, port=port)
        # Additional tools can be added here with their configurations
        else:
            raise ValueError(f"Unsupported monitoring tool: {tool}")

    def initialize_parameters(self, client_manager):
        """
        Initializes the parameters of the monitoring strategy.

        Args:
            client_manager: The client manager object.

        Returns:
            The initialized parameters.

        """
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, rnd, parameters, client_manager):
        """
        Configures the fit process for the monitoring strategy.

        Args:
            rnd (int): The random seed.
            parameters (dict): The parameters for the fit process.
            client_manager (ClientManager): The client manager.

        Returns:
            list: The selected clients for the fit process.
        """
        clients = self.base_strategy.configure_fit(rnd, parameters, client_manager)
        self.monitoring.observe_metric("clients_selected", len(clients))
        return clients

    def configure_evaluate(self, rnd, parameters, client_manager):
        """
        Configure the evaluation process.

        Args:
            rnd (int): The current round number.
            parameters (dict): The parameters for the evaluation.
            client_manager (ClientManager): The client manager for the evaluation.

        Returns:
            The configuration for the evaluation process.
        """
        return self.base_strategy.configure_evaluate(rnd, parameters, client_manager)

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregates the fitness results from multiple clients and returns the aggregated weights.

        Parameters:
        - rnd (int): The current round number.
        - results (list): A list of fitness results from the clients.
        - failures (list): A list of failed fitness results.

        Returns:
        - aggregated_weights: The aggregated weights obtained from the fitness results.

        """
        # Rest of the code...
        self.monitoring.track_resource_usage()
        start_communication = time.time()
        aggregated_weights = self.base_strategy.aggregate_fit(rnd, results, failures)
        communication_time = time.time() - start_communication
        self.monitoring.observe_metric("communication_time", communication_time)

        accuracies = [fit_res.metrics["accuracy"] for _, fit_res in results if "accuracy" in fit_res.metrics]
        training_times = [fit_res.metrics["training_time"] for _, fit_res in results if "training_time" in fit_res.metrics]

        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            self.monitoring.observe_metric("accuracy", avg_accuracy)

        if training_times:
            avg_training_time = sum(training_times) / len(training_times)
            self.monitoring.observe_metric("training_time", avg_training_time)

        self.monitoring.observe_metric("client_participation", len(results))

        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        """
        Aggregates the evaluation results and tracks resource usage.

        Args:
            rnd (int): The current round number.
            results (list): A list of evaluation results.
            failures (list): A list of evaluation failures.

        Returns:
            float: The loss value calculated by the base strategy.
        """


        self.monitoring.track_resource_usage()
        loss = self.base_strategy.aggregate_evaluate(rnd, results, failures)

        losses = [evaluate_res.metrics["loss"] for _, evaluate_res in results if "loss" in evaluate_res.metrics]
        if losses:
            avg_loss = sum(losses) / len(losses)
            self.monitoring.observe_metric("loss", avg_loss)

        return loss
