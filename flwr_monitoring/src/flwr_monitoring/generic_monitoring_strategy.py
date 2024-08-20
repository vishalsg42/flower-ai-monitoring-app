import flwr as fl
from flwr.common import Parameters, EvaluateRes, FitRes, Metrics
from typing import List, Tuple, Optional
import time
from .aggregate_functions import aggregate_fit_metrics, aggregate_evaluate_metrics

class GenericMonitoringStrategy(fl.server.strategy.Strategy):
    def __init__(self, base_strategy, monitoring_tool):
        """
        Initializes a GenericMonitoringStrategy object.

        Args:
            base_strategy: The base strategy object.
            monitoring_tool: The monitoring tool object.

        Returns:
            None
        """
        self.base_strategy = base_strategy
        self.monitoring = monitoring_tool
        self.current_epoch = 0

    def initialize_parameters(self, client_manager):
        """
        Initializes the parameters of the monitoring strategy.

        Args:
            client_manager: The client manager object.

        Returns:
            The initialized parameters.

        """
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Configures the fit process for the monitoring strategy.

        Args:
            server_round (int): The current server round.
            parameters (Parameters): The parameters for the fit process.
            client_manager: The client manager.

        Returns:
            List: The selected clients for the fit process.
        """
        clients = self.base_strategy.configure_fit(server_round, parameters, client_manager)
        self.monitoring.observe_metric("clients_selected", len(clients))
        self.current_epoch += 1
        self.monitoring.observe_metric("current_epoch", self.current_epoch)
        return clients

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        """
        Configure the evaluation process for the monitoring strategy.

        Args:
            server_round (int): The current server round.
            parameters (Parameters): The parameters for the evaluation.
            client_manager: The client manager for the evaluation.

        Returns:
            The result of the base strategy's configure_evaluate method.
        """
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List[BaseException]):
        """
        Aggregates the fit results from multiple clients and returns the aggregated weights and metrics.

        Args:
            server_round (int): The current server round.
            results (List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]): The fit results from the clients.
            failures (List[BaseException]): The list of exceptions raised during the fit process.

        Returns:
            Tuple: A tuple containing the aggregated weights and metrics.
        """
        self.monitoring.track_resource_usage()
        start_communication = time.time()
        aggregated_weights, fit_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        communication_time = time.time() - start_communication
        self.monitoring.observe_metric("flower_communication_time_seconds", communication_time)

        accuracies = [fit_res.metrics["accuracy"] for _, fit_res in results if "accuracy" in fit_res.metrics]
        training_times = [fit_res.metrics["training_time"] for _, fit_res in results if "training_time" in fit_res.metrics]

        if accuracies:
            print(accuracies)
            avg_accuracy = sum(accuracies) / len(accuracies)
            self.monitoring.observe_metric("flower_accuracy", avg_accuracy)

        if training_times:
            print(training_times)
            avg_training_time = sum(training_times) / len(training_times)
            self.monitoring.observe_metric("flower_training_time_seconds", avg_training_time)

        self.monitoring.observe_metric("flower_client_participation", len(results))

        # Aggregate fit metrics using the provided aggregation function
        aggregated_metrics = aggregate_fit_metrics([fit_res.metrics for _, fit_res in results])

        return aggregated_weights, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]], failures: List[BaseException]):
        """
        Aggregates the evaluation results from multiple clients and returns the aggregated loss and metrics.

        Args:
            server_round (int): The current server round.
            results (List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]]): A list of tuples containing the client proxy and the evaluation result.
            failures (List[BaseException]): A list of exceptions raised during evaluation.

        Returns:
            Tuple[float, Dict[str, float]]: A tuple containing the aggregated loss and metrics.
        """
        self.monitoring.track_resource_usage()
        loss, evaluate_metrics = self.base_strategy.aggregate_evaluate(server_round, results, failures)

        losses = [evaluate_res.loss for _, evaluate_res in results]
        if losses:
            avg_loss = sum(losses) / len(losses)
            self.monitoring.observe_metric("flower_loss", avg_loss)

        # Aggregate evaluate metrics using the provided aggregation function
        aggregated_metrics = aggregate_evaluate_metrics([evaluate_res.metrics for _, evaluate_res in results])

        return loss, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Metrics]]:
        """
        Evaluates the monitoring strategy for a given server round and parameters.

        Args:
            server_round (int): The current server round.
            parameters (Parameters): The parameters for the evaluation.

        Returns:
            Optional[Tuple[float, Metrics]]: A tuple containing the evaluation result as a float and the metrics as a Metrics object.
        """
        return self.base_strategy.evaluate(server_round, parameters)
