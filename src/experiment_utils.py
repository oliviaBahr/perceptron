import pathlib
from os import listdir
from os.path import basename, isdir, join
from typing import Any, Literal

import comet_ml
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from tqdm import trange

from implementation.loader import load_dir
from implementation.soup_classifier import SoupClassifier
from implementation.voted_classifier import VotedClassifier

classification_dir = join(pathlib.Path(__file__).parent.resolve(), "../classification-data")
DATA_DIRS = [join(classification_dir, d) for d in listdir(classification_dir)]
DATA_DIRS = [d for d in DATA_DIRS if isdir(d)]

ARCH_TYPE = Literal["perc", "svm", "logistic"]
ENSEMBLE_TYPE = Literal["none", "soup", "voted"]


def _make_model(
    arch: ARCH_TYPE,
    ensemble_type: ENSEMBLE_TYPE,
    max_epochs_per_learner: int | None,
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
    project_name: str,
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
        project_name=project_name,
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
