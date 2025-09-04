import os
import csv
import json
import datetime
from pathlib import Path

class MetricsLogger:
    def __init__(self, base_dir="logs"):
        timestamp = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M")
        self.log_dir = Path(base_dir) / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = self.log_dir / "metrics.csv"
        self.json_file = self.log_dir / "metrics.jsonl"

        self.csv_writer = None
        self.csv_fp = open(self.csv_file, "w", newline="")
        self.json_fp = open(self.json_file, "w")

    def log(self, metrics: dict):
        if self.csv_writer is None:
            fieldnames = list(metrics.keys())
            self.csv_writer = csv.DictWriter(self.csv_fp, fieldnames=fieldnames)
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
        self.csv_fp.flush()

        self.json_fp.write(json.dumps(metrics) + "\n")
        self.json_fp.flush()

    def close(self):
        self.csv_fp.close()
        self.json_fp.close()