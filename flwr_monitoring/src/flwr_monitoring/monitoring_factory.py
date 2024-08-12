from .prometheus_monitoring import PrometheusMonitoring
from .config import config

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
    if tool_name == "prometheus":
        return PrometheusMonitoring(metrics, tool_config)
    else:
        raise ValueError(f"Unknown monitoring tool: {tool_name}")
