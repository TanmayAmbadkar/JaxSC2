import os
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        """
        Simple TensorBoard logger for JAX metrics (using tensorboardX).
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log(self, step, metrics: dict):
        """
        Logs a dictionary of metrics at a given global step.
        """
        for k, v in metrics.items():
            try:
                self.writer.add_scalar(k, float(v), step)
            except (TypeError, ValueError):
                continue

    def close(self):
        self.writer.close()
