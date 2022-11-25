import jiant.proj.main.metarunner as jiant_metarunner
from jiant.utils.zlog import BaseZLogger, PRINT_LOGGER
import jiant.proj.main.runner as jiant_runner

class JiantMetarunner(jiant_metarunner.JiantMetarunner):
    def __init__(
        self,
        runner: jiant_runner.JiantRunner,
        save_every_steps,
        eval_every_steps,
        min_train_steps,
        save_checkpoint_every_steps,
        no_improvements_for_n_evals,
        checkpoint_saver,
        output_dir,
        verbose: bool = True,
        save_best_model: bool = True,
        load_best_model: bool = True,
        save_last_model: bool = True,
        log_writer: BaseZLogger = PRINT_LOGGER,
    ):
        super().__init__(
            runner,
            save_every_steps,
            eval_every_steps,
            save_checkpoint_every_steps,
            no_improvements_for_n_evals,
            checkpoint_saver,
            output_dir,
            verbose,
            save_best_model,
            load_best_model,
            save_last_model,
            log_writer,
        )
        self.min_train_steps = min_train_steps

    def should_break_training(self) -> bool:
        return super().should_break_training() and (self.min_train_steps == 0 or self.train_state.global_steps > self.min_train_steps)