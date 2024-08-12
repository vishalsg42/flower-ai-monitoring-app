from prometheus_client import Summary, Counter, Gauge, Histogram, start_http_server
from .base_monitoring_strategy import BaseMonitoringStrategy
import psutil
import GPUtil
import logging

class PrometheusMonitoring(BaseMonitoringStrategy):
    def __init__(self, metrics, config):
        super().__init__(metrics, config)
        self.metric_objects = self.initialize_metrics()
        self.server_port = config.get('port', 8000)
        self.server_url = config.get('url', '0.0.0.0')
        self.start_server()
        logging.basicConfig(level=logging.INFO)

    def initialize_metrics(self):
        metric_objects = {}
        for metric in self.metrics:
            metric_objects[metric["name"]] = metric["type"](
                metric["name"], metric["description"]
            )
        return metric_objects

    def track_resource_usage(self):
        if "cpu_usage" in self.metric_objects:
            cpu_usage = psutil.cpu_percent(interval=1)
            self.metric_objects["cpu_usage"].set(cpu_usage)

        if "memory_usage" in self.metric_objects:
            memory_info = psutil.virtual_memory()
            self.metric_objects["memory_usage"].set(memory_info.percent)

        if "gpu_usage" in self.metric_objects:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    self.metric_objects[f'gpu_{i}_usage'].set(gpu.load * 100)

    def observe_metric(self, metric_name, value):
        logging.info(f"Observing metric {metric_name} with value {value}")
        if metric_name in self.metric_objects:
            metric = self.metric_objects[metric_name]
            logging.info(f"Observing metric {metric} ")
            
            if isinstance(metric, Gauge):
                metric.set(value)
            elif isinstance(metric, (Summary, Histogram)):
                metric.observe(value)
            elif isinstance(metric, Counter):
                metric.inc(value)
            else:
                raise TypeError(f"Unsupported metric type: {type(metric)}")

    def start_server(self):
        start_http_server(self.server_port, addr=self.server_url)
