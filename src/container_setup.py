import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import jiant.proj.main.components.task_sampler as jiant_task_sampler
import jiant.shared.caching as caching
from jiant.tasks.core import Task
from jiant.tasks.retrieval import create_task_from_config_path
import jiant.utils.python.io as py_io
from jiant.utils.python.datastructures import ExtendedDataClassMixin

from jiant.proj.main.components.container_setup import *

def create_task_cache_dict(task_cache_config_dict: Dict) -> Dict:
    """Takes a map of task cache configs, and returns map of instantiated task data cache objects.
    Notes:
        This function assumes that data is divided and stored according to phase where phase takes
        a value of train, val, val_labels, or test.
    Args:
        task_cache_config_dict (Dict[str, Dict[str, str]]): maps of task names to cache file dirs.
    Returns:
        Dict[str, Dict[str, ChunkedFilesDataCache]] mappings from task name to task cache objects.
    """
    task_cache_dict = {}
    for task_name, task_cache_config in task_cache_config_dict.items():
        single_task_cache_dict = {}
        for phase in ["train", "val", "val_labels", "test", "test_labels", "train_labels"]:
            if phase in task_cache_config:
                single_task_cache_dict[phase] = caching.ChunkedFilesDataCache(
                    task_cache_config[phase],
                )
        task_cache_dict[task_name] = single_task_cache_dict
    return task_cache_dict

def create_jiant_task_container(
    task_config_path_dict: Dict,
    task_cache_config_dict: Dict,
    sampler_config: Dict,
    global_train_config: Dict,
    task_specific_configs_dict: Dict,
    metric_aggregator_config: Dict,
    taskmodels_config: Dict,
    task_run_config: Dict,
    verbose: bool = True,
) -> JiantTaskContainer:
    """Read and interpret config files, initialize configuration objects, return JiantTaskContainer.
    Args:
        task_config_path_dict (Dict[str, str]): map of task names to task config files.
        task_cache_config_dict (Dict[str, str]): map of task names to cache file dirs.
        sampler_config (Dict): map containing sample config options, e.g., uniform task sampling.
        global_train_config (Dict): map of training configs shared by all tasks (e.g., max_steps).
        task_specific_configs_dict (Dict): map of maps mapping task names to task-specific options.
        metric_aggregator_config (Dict): map containing task metric aggregation options.
        taskmodels_config: maps mapping from tasks to models, and specifying task-model configs.
        task_run_config: config determining which tasks are used in which phase (e.g., train).
        verbose: True to print task info.
    Returns:
        JiantTaskContainer carrying components configured and set up pre-runner.
    """
    task_dict = create_task_dict(task_config_dict=task_config_path_dict, verbose=verbose)
    task_cache_dict = create_task_cache_dict(task_cache_config_dict=task_cache_config_dict)
    global_train_config = GlobalTrainConfig.from_dict(global_train_config)
    task_specific_config = create_task_specific_configs(
        task_specific_configs_dict=task_specific_configs_dict,
    )
    taskmodels_config = TaskmodelsConfig.from_dict(taskmodels_config)
    task_run_config = TaskRunConfig.from_dict(task_run_config)

    num_train_examples_dict = get_num_train_examples(
        task_cache_dict=task_cache_dict, train_task_list=task_run_config.train_task_list,
    )
    task_sampler = jiant_task_sampler.create_task_sampler(
        sampler_config=sampler_config,
        # task sampler samples only from the training tasks
        task_dict={
            task_name: task_dict[task_name] for task_name in task_run_config.train_task_list
        },
        task_to_num_examples_dict=num_train_examples_dict,
    )
    metric_aggregator = jiant_task_sampler.create_metric_aggregator(
        metric_aggregator_config=metric_aggregator_config,
    )
    return JiantTaskContainer(
        task_dict=task_dict,
        task_sampler=task_sampler,
        global_train_config=global_train_config,
        task_cache_dict=task_cache_dict,
        task_specific_configs=task_specific_config,
        taskmodels_config=taskmodels_config,
        task_run_config=task_run_config,
        metrics_aggregator=metric_aggregator,
    )

def create_jiant_task_container_from_dict(
    jiant_task_container_config_dict: Dict, verbose: bool = True
) -> JiantTaskContainer:
    return create_jiant_task_container(
        task_config_path_dict=jiant_task_container_config_dict["task_config_path_dict"],
        task_cache_config_dict=jiant_task_container_config_dict["task_cache_config_dict"],
        sampler_config=jiant_task_container_config_dict["sampler_config"],
        global_train_config=jiant_task_container_config_dict["global_train_config"],
        task_specific_configs_dict=jiant_task_container_config_dict["task_specific_configs_dict"],
        taskmodels_config=jiant_task_container_config_dict["taskmodels_config"],
        task_run_config=jiant_task_container_config_dict["task_run_config"],
        metric_aggregator_config=jiant_task_container_config_dict["metric_aggregator_config"],
        verbose=verbose,
    )


def create_jiant_task_container_from_json(
    jiant_task_container_config_path: str, verbose: bool = True
) -> JiantTaskContainer:
    return create_jiant_task_container_from_dict(
        jiant_task_container_config_dict=py_io.read_json(jiant_task_container_config_path),
        verbose=verbose,
    )