from flwr.server.strategy import FedAvg
from typing import List, Tuple
from flwr.common import Metrics


def aggregate_fit_metrics(metrics: List[Metrics]) -> Metrics:
    """Aggregate fit metrics.

    This function takes a list of metrics and aggregates them to calculate the average accuracy and training time.

    Parameters:
    metrics (List[Metrics]): A list of metrics dictionaries.

    Returns:
    Metrics: A dictionary containing the aggregated metrics, including the average accuracy and training time.
    """
    accuracy = [m["accuracy"] for m in metrics if "accuracy" in m]
    training_time = [m["training_time"]
                     for m in metrics if "training_time" in m]
    return {
        "accuracy": sum(accuracy) / len(accuracy) if accuracy else 0.0,
        "training_time": sum(training_time) / len(training_time) if training_time else 0.0,
    }


def aggregate_evaluate_metrics(metrics: List[Metrics]) -> Metrics:
    """Aggregate evaluation metrics.

    Args:
        metrics (List[Metrics]): A list of evaluation metrics.

    Returns:
        Metrics: The aggregated evaluation metrics.

    """
    loss = [m["loss"] for m in metrics if "loss" in m]
    return {
        "loss": sum(loss) / len(loss) if loss else 0.0,
    }


class CustomFedAvg(FedAvg):
    """
    Custom implementation of the FedAvg strategy.

    Args:
        fit_metrics_aggregation_fn (callable): A function to aggregate the fit metrics.
        evaluate_metrics_aggregation_fn (callable): A function to aggregate the evaluate metrics.
    """
    def __init__(self):
        super().__init__(
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
