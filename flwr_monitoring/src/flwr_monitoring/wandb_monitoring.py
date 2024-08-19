from .base_monitoring_strategy import BaseMonitoringStrategy
import psutil
import GPUtil
import wandb
import logging


class WandBMonitoring(BaseMonitoringStrategy):
    def __init__(self, metrics, config):
        super().__init__(metrics, config)
        # Initialize WandB here
        wandb.init(project=config["project"], entity=config.get("entity"))
        self.metric_objects = self.initialize_metrics()
        logging.basicConfig(level=logging.INFO)

    def initialize_metrics(self):
        # In WandB, metrics are logged directly, so this might not be necessary,
        # but we can still maintain a structure to track what metrics are available.
        metric_objects = {}
        for metric in self.metrics:
            metric_objects[metric["name"]] = metric["type"](
                metric["name"], metric["description"]
            )
        return metric_objects

    def track_resource_usage(self):
        resource_metrics = {}
        if "cpu_usage" in self.metric_objects:
            cpu_usage = psutil.cpu_percent(interval=1)
            resource_metrics["cpu_usage"] = cpu_usage

        if "memory_usage" in self.metric_objects:
            memory_info = psutil.virtual_memory()
            resource_metrics["memory_usage"] = memory_info.percent

        if "gpu_usage" in self.metric_objects:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    resource_metrics[f'gpu_{i}_usage'] = gpu.load * 100

        # Log all resource metrics to WandB
        if resource_metrics:
            wandb.log(resource_metrics)

    def observe_metric(self, metric_name, value):
        logging.info(f"Observing metric {metric_name} with value {value}")
        # Directly log the metric to WandB
        if metric_name:
            wandb.log({metric_name: value})

    def start_server(self):
        # WandB does not require starting a server, so we can just pass here.
        logging.info("WandB monitoring initialized - no server to start.")
        pass