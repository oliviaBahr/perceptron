from os import listdir
from os.path import abspath, basename, isdir, join
from typing import Any, Literal, get_args

import comet_ml
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from tqdm import trange

from loader import load_dir
from soup_classifier import SoupClassifier
from voted_classifier import VotedClassifier

DATA_DIRS = [abspath(join("classification-data", d)) for d in listdir("classification-data")]
DATA_DIRS = [d for d in DATA_DIRS if isdir(d)]

ARCH_TYPE = Literal["perc", "svm", "logistic", "mlp"]
ENSEMBLE_TYPE = Literal["none", "soup", "voted"]


def _make_model(
    arch: ARCH_TYPE,
    ensemble_type: ENSEMBLE_TYPE,
    max_epochs_per_learner: int,
    num_learners: int,
    training_size: float,
    learner_kwargs: dict[str, Any],
):
    # 1. Determine appropriate kwargs for the learner
    LearnerClass = SGDClassifier if arch != "mlp" else MLPClassifier
    kwargs: dict[str, Any] = {"max_iter": max_epochs_per_learner}
    match arch:
        case "perc":
            kwargs.update(
                {
                    "loss": "perceptron",
                    "learning_rate": "constant",
                    "eta0": 1,
                    "penalty": None,
                }
            )
        case "svm":
            kwargs.update({"loss": "hinge", "penalty": None})
        case "logistic":
            kwargs.update({"loss": "log_loss", "penalty": None})
        case "mlp":
            pass
        case _:
            raise ValueError()

    # 2. Initialize the appropriate ensembled model
    if ensemble_type == "none":
        return LearnerClass(**kwargs, **learner_kwargs)
    elif ensemble_type == "soup":
        return SoupClassifier(
            n_learners=num_learners,
            training_size=training_size,
            LearnerClass=LearnerClass,  # type:ignore
            learner_kwargs=kwargs,
        )
    elif ensemble_type == "voted":
        return VotedClassifier(
            n_learners=num_learners,
            training_size=training_size,
            LearnerClass=LearnerClass,  # type:ignore
            learner_kwargs=kwargs,
        )


def run_experiment(
    n_runs: int,
    dirpath: str,
    arch: ARCH_TYPE,
    ensemble_type: ENSEMBLE_TYPE,
    max_epochs_per_learner: int,
    num_learners: int,
    training_size: float,
    learner_kwargs: dict[str, Any],
):
    """Runs experiments with a given model and dataset

    Args:
        n_runs (int): The number of repeated runs to perform
        dirpath (str): Absolute path to the directory containing the dataset
    """
    experiment = comet_ml.Experiment(
        project_name="baselines-v3",
        workspace="perceptrons",
        log_code=False,
        log_graph=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
    )
    experiment.log_parameters(
        {
            "dataset": basename(dirpath),
            "arch": arch,
            "ensemble_type": ensemble_type,
            "max_epochs_per_learner": max_epochs_per_learner,
            "num_learners": num_learners,
            "training_size": training_size,
            "learner_kwargs": learner_kwargs,
        }
    )

    (X, y), (tX, ty) = load_dir(dirpath)

    for _ in trange(n_runs, desc="Running experiments"):
        with experiment.train():
            X, y = shuffle(X, y)  # type:ignore
            model = _make_model(
                arch=arch,
                ensemble_type=ensemble_type,
                max_epochs_per_learner=max_epochs_per_learner,
                num_learners=num_learners,
                training_size=training_size,
                learner_kwargs=learner_kwargs,
            )
            model.fit(X, y)

        with experiment.test():
            experiment.log_metric("accuracy", model.score(tX, ty))
    experiment.end()


if __name__ == "__main__":
    comet_ml.login()

    NUM_RUNS = 100
    NUM_ITERATIONS = 100  # num epochs for single models, num learners for ensembles
    NUM_EPOCHS_PER_LEARNER = 1
    TRAINING_SIZE = 0.5

    for dirpath in DATA_DIRS:
        for arch in get_args(ARCH_TYPE):
            # Skip the MLP for baselines, let's make that a separate study
            if "mlp" in arch:
                continue

            for ensemble_type in get_args(ENSEMBLE_TYPE):
                print(f"Training {arch} - {ensemble_type}")

                run_experiment(
                    n_runs=NUM_RUNS,
                    dirpath=dirpath,
                    arch=arch,
                    ensemble_type=ensemble_type,
                    max_epochs_per_learner=(NUM_ITERATIONS if ensemble_type == "none" else NUM_EPOCHS_PER_LEARNER),
                    num_learners=NUM_ITERATIONS,
                    training_size=TRAINING_SIZE,
                    learner_kwargs={},
                )
