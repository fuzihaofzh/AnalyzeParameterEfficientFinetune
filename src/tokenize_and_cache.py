import os

from transformers import AutoConfig
from transformers import AutoTokenizer

import jiant.proj.main.preprocessing as preprocessing
import jiant.shared.caching as shared_caching
import jiant.tasks.evaluate as evaluate
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf

from jiant.proj.main.modeling.primary import JiantTransformersModelFactory
from jiant.shared.constants import PHASE
from jiant.tasks.retrieval import create_task_from_config_path

from jiant.proj.main.tokenize_and_cache import *

def main(args: RunConfiguration):
    config = AutoConfig.from_pretrained(args.hf_pretrained_model_name_or_path)
    model_type = config.model_type

    task = create_task_from_config_path(config_path=args.task_config_path, verbose=True)
    feat_spec = JiantTransformersModelFactory.build_featurization_spec(
        model_type=model_type, max_seq_length=args.max_seq_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_pretrained_model_name_or_path, use_fast=False)
    if isinstance(args.phases, str):
        phases = args.phases.split(",")
    else:
        phases = args.phases
    #assert set(phases) <= {PHASE.TRAIN, PHASE.VAL, PHASE.TEST}
    #maple remove

    paths_dict = {}
    os.makedirs(args.output_dir, exist_ok=True)

    if PHASE.TRAIN in phases:
        chunk_and_save(
            task=task,
            phase=PHASE.TRAIN,
            examples=task.get_train_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
        paths_dict["train"] = os.path.join(args.output_dir, PHASE.TRAIN)

    if PHASE.VAL in phases:
        val_examples = task.get_val_examples()
        chunk_and_save(
            task=task,
            phase=PHASE.VAL,
            examples=val_examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
        evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task)
        shared_caching.chunk_and_save(
            data=evaluation_scheme.get_labels_from_cache_and_examples(
                task=task,
                cache=shared_caching.ChunkedFilesDataCache(
                    os.path.join(args.output_dir, PHASE.VAL)
                ),
                examples=val_examples,
            ),
            chunk_size=args.chunk_size,
            data_args=args.to_dict(),
            output_dir=os.path.join(args.output_dir, "val_labels"),
        )
        paths_dict[PHASE.VAL] = os.path.join(args.output_dir, PHASE.VAL)
        paths_dict["val_labels"] = os.path.join(args.output_dir, "val_labels")

    if PHASE.TEST in phases:
        chunk_and_save(
            task=task,
            phase=PHASE.TEST,
            examples=task.get_test_examples(),
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
        paths_dict[PHASE.TEST] = os.path.join(args.output_dir, PHASE.TEST)

    if "test_labels" in phases:
        from jiant.utils.python.io import read_jsonl
        test_examples = task._create_examples(lines=read_jsonl(task.test_path), set_type="val")
        chunk_and_save(
            task=task,
            phase=PHASE.TEST,
            examples=test_examples,
            feat_spec=feat_spec,
            tokenizer=tokenizer,
            args=args,
        )
        evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task)
        shared_caching.chunk_and_save(
            data=evaluation_scheme.get_labels_from_cache_and_examples(
                task=task,
                cache=shared_caching.ChunkedFilesDataCache(
                    os.path.join(args.output_dir, PHASE.TEST)
                ),
                examples=test_examples,
            ),
            chunk_size=args.chunk_size,
            data_args=args.to_dict(),
            output_dir=os.path.join(args.output_dir, "test_labels"),
        )
        paths_dict["test_labels"] = os.path.join(args.output_dir, "test_labels")
        #===for train labels:
        evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task)
        shared_caching.chunk_and_save(
            data=evaluation_scheme.get_labels_from_cache_and_examples(
                task=task,
                cache=shared_caching.ChunkedFilesDataCache(
                    os.path.join(args.output_dir, PHASE.TRAIN)
                ),
                examples=task.get_train_examples(),
            ),
            chunk_size=args.chunk_size,
            data_args=args.to_dict(),
            output_dir=os.path.join(args.output_dir, "train_labels"),
        )
        paths_dict["train_labels"] = os.path.join(args.output_dir, "train_labels")

    if not args.skip_write_output_paths:
        py_io.write_json(data=paths_dict, path=os.path.join(args.output_dir, "paths.json"))


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())