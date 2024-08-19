from .prometheus_monitoring import PrometheusMonitoring
from .config import config
from .wandb_monitoring import WandBMonitoring  # Import the WandB monitoring class

def create_monitoring_tool(tool_name, metrics, config={}):
    """
    Creates a monitoring tool based on the given tool_name and metrics.

    Parameters:
    - tool_name (str): The name of the monitoring tool.
    - metrics (list): A list of metrics to be monitored.

    Returns:
    - monitoring_tool: An instance of the monitoring tool.

    Raises:
    - ValueError: If the given tool_name is unknown.
    """
    tool_config = config.get(tool_name, config)
    # print("Creating monitoring tool", tool_name, tool_config)
    if tool_name == "prometheus":
        return PrometheusMonitoring(metrics, tool_config)
    elif tool_name == "wandb":
        print("Creating WandB monitoring tool", tool_config)
        return WandBMonitoring(metrics, tool_config)
    else:
        raise ValueError(f"Unknown monitoring tool: {tool_name}")
