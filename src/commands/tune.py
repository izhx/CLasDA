"""
Modified from https://github.com/himkt/allennlp-optuna/blob/main/allennlp_optuna/commands/tune.py
"""

import argparse
import json
import logging
import os
import shutil

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import evaluate_file, _environment_variables
import optuna
from optuna import Trial
from optuna.integration import AllenNLPExecutor

logger = logging.getLogger(__name__)


def tune(args: argparse.Namespace) -> None:
    os.makedirs(args.serialization_dir, exist_ok=True)

    if args.optuna_path is not None and os.path.isfile(args.optuna_path):
        optuna_config = json.loads(evaluate_file(
            args.optuna_path, ext_vars=_environment_variables())
        )
    else:
        raise RuntimeError(f"<{args.optuna_path}> is not a optuna config file!")

    def objective(trial: Trial) -> float:

        for hparam in optuna_config["hparams"]:
            attr_type = hparam["type"]
            suggest = getattr(trial, "suggest_{}".format(attr_type))
            suggest(**hparam["attributes"])

        optuna_serialization_dir = os.path.join(
            args.serialization_dir, "trial_{}".format(trial.number)
        )
        # hack for checkpointer in config_file
        os.environ['ALLENNLP_SERIALIZATION_DIR'] = optuna_serialization_dir
        executor = AllenNLPExecutor(
            trial=trial,  # trial object
            config_file=args.param_path,  # path to jsonnet
            serialization_dir=optuna_serialization_dir,
            metrics=args.metrics,
            include_package=args.include_package,
            # force=True,
            file_friendly_logging=True
        )
        return executor.run()

    if "pruner" in optuna_config:
        pruner_class = getattr(optuna.pruners, optuna_config["pruner"]["type"])
        pruner = pruner_class(**optuna_config["pruner"].get("attributes", {}))
    else:
        pruner = None

    if "sampler" in optuna_config:
        sampler_class = getattr(optuna.samplers, optuna_config["sampler"]["type"])
        sampler = sampler_class(optuna_config["sampler"].get("attributes", {}))
    else:
        sampler = None

    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        storage=args.storage,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=args.load_if_exists,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,  # number of trials to train a model
        timeout=args.timeout,  # threshold for executing time (sec)
        n_jobs=args.n_jobs,  # number of processes in parallel execution
    )

    print("\n\nNumber of finished trials: ", len(study.trials))
    trial = study.best_trial
    print("Best trial:", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_trial_dir = f"{args.serialization_dir}/trial_{trial.number}"
    best_archive_dir = args.serialization_dir + "_best"
    shutil.copytree(best_trial_dir, best_archive_dir)
    shutil.rmtree(args.serialization_dir, ignore_errors=True)
    print(f"\nSaved best AllenNLP archives to `{best_archive_dir}`.")


@Subcommand.register("tune")
class Tune(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Optimize hyperparameter of a model.")

        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained",
        )

        subparser.add_argument(
            "optuna_path",
            type=str,
            help="path to optuna config file",
            default="hyper_params.json",
        )

        subparser.add_argument(
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        # ---- Optuna -----

        subparser.add_argument(
            "--load-if-exists",
            default=False,
            action="store_true",
            help="If specified, the creation of the study is skipped "
            "without any error when the study name is duplicated.",
        )

        subparser.add_argument(
            "--direction",
            type=str,
            choices=("minimize", "maximize"),
            default="minimize",
            help="Set direction of optimization to a new study. Set 'minimize' "
            "for minimization and 'maximize' for maximization.",
        )

        subparser.add_argument(
            "--n-trials",
            type=int,
            help="The number of trials. If this argument is not given, as many " "trials run as possible.",
            default=50,
        )

        subparser.add_argument(
            "--n-jobs",
            type=int,
            help="The number of parallel jobs. If this argument is set to :obj:`-1`, the number is set to CPU count.",
            default=1,
        )

        subparser.add_argument(
            "--timeout",
            type=float,
            help="Stop study after the given number of second(s). If this argument"
            " is not given, as many trials run as possible.",
        )

        subparser.add_argument(
            "--study-name", default=None, help="The name of the study to start optimization on."
        )

        subparser.add_argument(
            "--storage",
            type=str,
            help=(
                "The path to storage. "
                "allennlp-optuna supports a valid URL" "for sqlite3, mysql, postgresql, or redis."
            ),
            default="sqlite:///allennlp_optuna.db",
        )

        subparser.add_argument(
            "--metrics",
            type=str,
            help="The metrics you want to optimize.",
            default="best_validation_loss",
        )

        subparser.set_defaults(func=tune)
        return subparser
