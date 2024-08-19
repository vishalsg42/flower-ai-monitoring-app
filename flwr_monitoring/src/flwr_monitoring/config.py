config = {
    """
    Configuration settings for flower-ai monitoring.

    Attributes:
        prometheus (dict): Configuration settings for Prometheus.
            - port (int): The port number for Prometheus server.
            - url (str): The URL for Prometheus server.
            - push_interval (int): The interval in seconds for pushing metrics to Prometheus.

    """
    "prometheus": {
        "port": 8000,
        "url": "0.0.0.0",
        "push_interval": 15  # Seconds
    },
    "wandb": {
        "project": "my-awesome-project",
    }
    # Additional configurations for other tools can be added here
}
