from flwr_monitoring.model import Net
import flwr as fl
from flwr_monitoring import GenericMonitoringStrategy, default_metrics, create_monitoring_tool, aggregate_fit_metrics, aggregate_evaluate_metrics
import os
import sys

# Ensure the src directory and examples/prometheus directory are in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../src'))
example_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, root_dir)
sys.path.insert(0, example_dir)


def main():
    # Initialize the base model
    model = Net()

    # Convert the model to parameters
    model_params = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Initialize the base strategy
    base_strategy = fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(model_params),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    # Create the monitoring tool instance for Prometheus
    monitoring_tool_instance = create_monitoring_tool(
        tool_name="prometheus",
        metrics=default_metrics,
        config={"port": 8000, "url": "0.0.0.0"}
    )

    # Wrap the base strategy with the monitoring strategy
    monitoring_strategy = GenericMonitoringStrategy(
        base_strategy, monitoring_tool_instance)

    # Start the Flower server with the custom strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=monitoring_strategy,
    )


if __name__ == "__main__":
    main()
