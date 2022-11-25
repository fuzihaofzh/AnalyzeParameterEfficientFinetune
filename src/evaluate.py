import json
import os

import torch

import jiant.utils.python.io as py_io
import jiant.proj.main.components.task_sampler as jiant_task_sampler

from jiant.proj.main.components.evaluate import *

def write_val_results(val_results_dict, metrics_aggregator, output_dir, verbose=True, result_file = "val_metrics.json"):
    full_results_to_write = {
        "aggregated": jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=metrics_aggregator, results_dict=val_results_dict,
        ),
    }
    for task_name, task_results in val_results_dict.items():
        task_results_to_write = {}
        if "loss" in task_results:
            task_results_to_write["loss"] = task_results["loss"]
        if "metrics" in task_results:
            task_results_to_write["metrics"] = task_results["metrics"].to_dict()
        full_results_to_write[task_name] = task_results_to_write

    metrics_str = json.dumps(full_results_to_write, indent=2)
    if verbose:
        print(metrics_str)

    py_io.write_json(data=full_results_to_write, path=os.path.join(output_dir, result_file))
    print("Saved at " + os.path.join(output_dir, result_file))