import os
import torch
import datetime


import jiant.proj.main.modeling.model_setup as jiant_model_setup
import runner as jiant_runner
#import jiant.proj.main.components.container_setup as container_setup
#import jiant.proj.main.metarunner as jiant_metarunner
import metarunner as jiant_metarunner
#import jiant.proj.main.components.evaluate as jiant_evaluate
import evaluate as jiant_evaluate
import jiant.shared.initialization as initialization
import jiant.shared.distributed as distributed
#import jiant.shared.model_setup as model_setup
import model_setup
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf
import zlog # maple
import container_setup # maple


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    jiant_task_container_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    hf_pretrained_model_name_or_path = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_transformers", type=str)

    # === Running Setup === #
    do_train = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")
    do_save = zconf.attr(action="store_true")
    do_save_last = zconf.attr(action="store_true")
    do_save_best = zconf.attr(action="store_true")
    write_val_preds = zconf.attr(action="store_true")
    write_test_preds = zconf.attr(action="store_true")
    eval_every_steps = zconf.attr(type=int, default=0)
    min_train_steps = zconf.attr(type=int, default=0)# maple
    save_every_steps = zconf.attr(type=int, default=0)
    save_checkpoint_every_steps = zconf.attr(type=int, default=0)
    no_improvements_for_n_evals = zconf.attr(type=int, default=0)
    keep_checkpoint_when_done = zconf.attr(action="store_true")
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)


@zconf.run_config
class ResumeConfiguration(zconf.RunConfig):
    checkpoint_path = zconf.attr(type=str)


def setup_runner(
    args: RunConfiguration,
    jiant_task_container: container_setup.JiantTaskContainer,
    quick_init_out,
    verbose: bool = True,
) -> jiant_runner.JiantRunner:
    """Setup jiant model, optimizer, and runner, and return runner.

    Args:
        args (RunConfiguration): configuration carrying command line args specifying run params.
        jiant_task_container (container_setup.JiantTaskContainer): task and sampler configs.
        quick_init_out (QuickInitContainer): device (GPU/CPU) and logging configuration.
        verbose: If True, enables printing configuration info (to standard out).

    Returns:
        jiant_runner.JiantRunner

    """
    # TODO document why the distributed.only_first_process() context manager is being used here.
    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        jiant_model = jiant_model_setup.setup_jiant_model(
            hf_pretrained_model_name_or_path=args.hf_pretrained_model_name_or_path,
            model_config_path=args.model_config_path,
            task_dict=jiant_task_container.task_dict,
            taskmodels_config=jiant_task_container.taskmodels_config,
        )
        jiant_model_setup.delegate_load_from_path(
            jiant_model=jiant_model, weights_path=args.model_path, load_mode=args.model_load_mode
        )
        jiant_model.to(quick_init_out.device)

    user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None for e in (args.user_mode[0].split(',') if type(args.user_mode) is not str else args.user_mode.split(',')) }
    if 'sctmask' in user_mode:
        import numpy as np
        parms = dict(jiant_model.named_parameters())
        max_len = max([parms[n].numel() for n in parms])
        max_ids = list(range(max_len))
        jiant_model.encoder.idx_dict = {}
        for name in parms:
            if not name.startswith('encoder'):
                continue
            fpar = parms[name].view(-1)
            xname = "sctmask__" + name.replace(".", "__")
            idx = torch.tensor(np.random.choice(max_ids[:len(fpar)], size = int(len(fpar) * float(user_mode['sctmask'])), replace = False)).to(fpar.device)
            x = fpar[idx].detach()
            x.requires_grad = True
            jiant_model.encoder.idx_dict[xname] = idx
            setattr(jiant_model.encoder, xname, torch.nn.Parameter(x))
            x = getattr(jiant_model.encoder, xname)
            parms[name].requires_grad = False
            parms[name].flatten()[idx] = x
    elif 'embtune' in user_mode:
        parms = dict(jiant_model.named_parameters())
        for p in parms:
            if 'embeddings' not in p and 'taskmodels_dict' not in p:
                parms[p].requires_grad = False
    elif 'prompt' in user_mode:
        parms = dict(jiant_model.named_parameters())
        for name in parms:
            if name.startswith('encoder'):
                parms[name].requires_grad = False
        jiant_model.encoder.embeddings.prompt_weight = torch.nn.Parameter(jiant_model.encoder.embeddings.word_embeddings.weight.new_zeros([int(user_mode['prompt']), 768]))
        torch.nn.init.xavier_uniform_(jiant_model.encoder.embeddings.prompt_weight)
        jiant_model.encoder.embeddings.prompt_weight.requires_grad = True
        jiant_model.encoder.embeddings.word_embeddings.weight_ori = jiant_model.encoder.embeddings.word_embeddings.weight
        jiant_model.encoder.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.cat([jiant_model.encoder.embeddings.word_embeddings.weight_ori, jiant_model.encoder.embeddings.prompt_weight]))
    elif 'diffprun' in user_mode:
        parms = dict(jiant_model.named_parameters())
        jiant_model.ori_pars = {}
        for p in parms:
            w = torch.zeros_like(parms[p])
            wname = "w__" + p.replace(".", "__")
            setattr(jiant_model.encoder, wname, torch.nn.Parameter(w))
            bername = "ber__" + p.replace(".", "__")
            ber = torch.randn_like(parms[p]).sigmoid()
            setattr(jiant_model.encoder, bername, torch.nn.Parameter(ber))
            jiant_model.ori_pars[p] = parms[p].data
            parms[p].requires_grad = False
            parms[p].detach_()
    elif 'adapter' in user_mode:
        jiant_model.encoder.add_adapter("adapter")
        jiant_model.encoder.train_adapter("adapter")
        jiant_model.encoder.to(jiant_model.encoder.device)
    elif 'lora' in user_mode:
        import loralib as lora
        import math
        import torch.nn as nn
        r = int(user_mode['lora']) if ('lora' in user_mode) else 16
        def set_lora(attn , name):
            linear = getattr(attn, name)
            q = lora.Linear(linear.in_features, linear.out_features, r).to(linear.weight.device)
            q.weight.data[:] = linear.weight.data[:]
            q.bias.data[:] = linear.bias.data[:]
            nn.init.kaiming_uniform_(q.lora_B, a=math.sqrt(5))
            setattr(attn, name, q)
        for i in range(len(jiant_model.encoder.encoder.layer)):
            set_lora(jiant_model.encoder.encoder.layer[i].attention.self, 'query')
            set_lora(jiant_model.encoder.encoder.layer[i].attention.self, 'key')
            set_lora(jiant_model.encoder.encoder.layer[i].attention.self, 'value')
            set_lora(jiant_model.encoder.encoder.layer[i].attention.output, 'dense')
            set_lora(jiant_model.encoder.encoder.layer[i].intermediate, 'dense')
            set_lora(jiant_model.encoder.encoder.layer[i].output, 'dense')
        lora.mark_only_lora_as_trainable(jiant_model)
    
    optimizer_scheduler = model_setup.create_optimizer(
        model=jiant_model,
        learning_rate=args.learning_rate,
        t_total=jiant_task_container.global_train_config.max_steps,
        warmup_steps=jiant_task_container.global_train_config.warmup_steps,
        warmup_proportion=None,
        optimizer_type=args.optimizer_type,
        verbose=verbose,
    )
    jiant_model, optimizer = model_setup.raw_special_model_setup(
        model=jiant_model,
        optimizer=optimizer_scheduler.optimizer,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        n_gpu=quick_init_out.n_gpu,
        local_rank=args.local_rank,
    )
    optimizer_scheduler.optimizer = optimizer
    rparams = jiant_runner.RunnerParameters(
        local_rank=args.local_rank,
        n_gpu=quick_init_out.n_gpu,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
    )
    runner = jiant_runner.JiantRunner(
        jiant_task_container=jiant_task_container,
        jiant_model=jiant_model,
        optimizer_scheduler=optimizer_scheduler,
        device=quick_init_out.device,
        rparams=rparams,
        log_writer=quick_init_out.log_writer,
    )
    runner.user_mode = user_mode
    if 'mixout' in user_mode:
        import copy
        runner.encoder0 = copy.deepcopy(jiant_model.encoder)
    return runner


def run_loop(args: RunConfiguration, checkpoint=None):
    is_resumed = checkpoint is not None
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    # maple
    quick_init_out.log_writer = zlog.ZLogger(os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d%H%m%S")), overwrite=True)
    print(quick_init_out.n_gpu)
    with quick_init_out.log_writer.log_context():
        jiant_task_container = container_setup.create_jiant_task_container_from_json(
            jiant_task_container_config_path=args.jiant_task_container_config_path, verbose=True,
        )
        runner = setup_runner(
            args=args,
            jiant_task_container=jiant_task_container,
            quick_init_out=quick_init_out,
            verbose=True,
        )
        if is_resumed:
            runner.load_state(checkpoint["runner_state"])
            del checkpoint["runner_state"]
        checkpoint_saver = jiant_runner.CheckpointSaver(
            metadata={"args": args.to_dict()},
            save_path=os.path.join(args.output_dir, "checkpoint.p"),
        )
        if args.do_train:
            metarunner = jiant_metarunner.JiantMetarunner(
                runner=runner,
                save_every_steps=args.save_every_steps,
                eval_every_steps=args.eval_every_steps,
                min_train_steps = args.min_train_steps,
                save_checkpoint_every_steps=args.save_checkpoint_every_steps,
                no_improvements_for_n_evals=args.no_improvements_for_n_evals,
                checkpoint_saver=checkpoint_saver,
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save or args.do_save_best,
                save_last_model=args.do_save or args.do_save_last,
                load_best_model=True,
                log_writer=quick_init_out.log_writer,
            )
            if is_resumed:
                metarunner.load_state(checkpoint["metarunner_state"])
                del checkpoint["metarunner_state"]
            metarunner.run_train_loop()

        if args.do_val:
            val_results_dict = runner.run_val(
                task_name_list=runner.jiant_task_container.task_run_config.val_task_list,
                return_preds=args.write_val_preds,
            )
            jiant_evaluate.write_val_results(
                val_results_dict=val_results_dict,
                metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                output_dir=args.output_dir,
                verbose=True,
            )
            if args.write_val_preds:
                jiant_evaluate.write_preds(
                    eval_results_dict=val_results_dict,
                    path=os.path.join(args.output_dir, "val_preds.p"),
                )
        else:
            assert not args.write_val_preds

        if args.do_test:#maple
            test_results_dict = runner.run_val(
                task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
                return_preds=False,
                phase = "test"
            )
            jiant_evaluate.write_val_results(
                val_results_dict=test_results_dict,
                metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                output_dir=args.output_dir,
                verbose=True,
                result_file = "test_metrics.json"
            )
            train_results_dict = runner.run_val(
                task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
                return_preds=False,
                phase = "train"
            )
            jiant_evaluate.write_val_results(
                val_results_dict=train_results_dict,
                metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                output_dir=args.output_dir,
                verbose=True,
                result_file = "train_metrics.json"
            )

        if args.write_test_preds:
            test_results_dict = runner.run_test(
                task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
            )
            jiant_evaluate.write_preds(
                eval_results_dict=test_results_dict,
                path=os.path.join(args.output_dir, "test_preds.p"),
            )

    if (
        not args.keep_checkpoint_when_done
        and args.save_checkpoint_every_steps
        and os.path.exists(os.path.join(args.output_dir, "checkpoint.p"))
    ):
        os.remove(os.path.join(args.output_dir, "checkpoint.p"))

    py_io.write_file("DONE", os.path.join(args.output_dir, "done_file"))


def run_resume(args: ResumeConfiguration):
    resume(checkpoint_path=args.checkpoint_path)


def resume(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    args = RunConfiguration.from_dict(checkpoint["metadata"]["args"])
    run_loop(args=args, checkpoint=checkpoint)


def run_with_continue(cl_args):
    run_args = RunConfiguration.default_run_cli(cl_args=cl_args)
    if not run_args.force_overwrite and (
        os.path.exists(os.path.join(run_args.output_dir, "done_file"))
        or os.path.exists(os.path.join(run_args.output_dir, "val_metrics.json"))
    ):
        print("Already Done")
        return
    elif run_args.save_checkpoint_every_steps and os.path.exists(
        os.path.join(run_args.output_dir, "checkpoint.p")
    ):
        print("Resuming")
        resume(os.path.join(run_args.output_dir, "checkpoint.p"))
    else:
        print("Running from start")
        run_loop(args=run_args)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "run":
        run_loop(RunConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "continue":
        run_resume(ResumeConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "run_with_continue":
        run_with_continue(cl_args=cl_args)
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
