from typing import List
from flwr.common import Metrics

def aggregate_fit_metrics(metrics: List[Metrics]) -> Metrics:
    """Aggregate fit metrics.

    Aggregates the fit metrics by calculating the average accuracy and training time.

    Args:
        metrics (List[Metrics]): A list of dictionaries containing the metrics.

    Returns:
        Metrics: A dictionary containing the aggregated metrics with keys 'accuracy' and 'training_time'.

    """
    accuracy = [m["accuracy"] for m in metrics if "accuracy" in m]
    training_time = [m["training_time"] for m in metrics if "training_time" in m]
    return {
        "accuracy": sum(accuracy) / len(accuracy) if accuracy else 0.0,
        "training_time": sum(training_time) / len(training_time) if training_time else 0.0,
    }

def aggregate_evaluate_metrics(metrics: List[Metrics]) -> Metrics:
    """Aggregate evaluation metrics.

    This function takes a list of metrics and aggregates them to calculate the average loss.

    Parameters:
        metrics (List[Metrics]): A list of metrics dictionaries.

    Returns:
        Metrics: A dictionary containing the aggregated metrics.

    """
    loss = [m["loss"] for m in metrics if "loss" in m]
    return {
        "loss": sum(loss) / len(loss) if loss else 0.0,
    }
