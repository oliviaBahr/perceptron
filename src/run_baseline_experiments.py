import inspect
from os import listdir
from os.path import abspath, basename, join
from typing import Any, Literal, get_args

import comet_ml
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from tqdm import trange

import addmlpclassifier
import addsgdclassifier
from loader import load_dir

MODEL_TYPE = Literal[
    "perc",
    "perc_soup",
    "svm",
    "svm_soup",
    "logistic",
    "logistic_soup",
    "mlp",
    "mlp_soup",
]

CLASSIFIERS: dict[MODEL_TYPE, tuple[Any, dict[str, str | int | None]]] = {
    "perc": (Perceptron, {}),
    "perc_soup": (
        addsgdclassifier.AddSGDClassifier,
        {"loss": "perceptron", "learning_rate": "constant", "eta0": 1, "penalty": None},
    ),
    "svm": (SGDClassifier, {"loss": "hinge", "penalty": None}),
    "svm_soup": (addsgdclassifier.AddSGDClassifier, {"loss": "hinge", "penalty": None}),
    "logistic": (SGDClassifier, {"loss": "log_loss", "penalty": None}),
    "logistic_soup": (
        addsgdclassifier.AddSGDClassifier,
        {"loss": "log_loss", "penalty": None},
    ),
    "mlp": (MLPClassifier, {}),
    "mlp_soup": (addmlpclassifier.AddMLPClassifier, {}),
}

DATA_DIRS = [abspath(join("classification-data", d)) for d in listdir("classification-data")]


def run_experiment(
    n_runs: int,
    model_type: MODEL_TYPE,
    dirpath: str,
):
    """Runs experiments with a given model and dataset

    Args:
        n_runs (int): The number of repeated runs to perform
        model_type (Literal['perc', 'add_perc', 'svm', 'add_svm', 'lr', 'add_lr', 'mlp', 'add_mlp']): The type of model to train
        dirpath (str): Absolute path to the directory containing the dataset
    """
    (X, y), (tX, ty) = load_dir(dirpath)
    ModelClass, kwargs = CLASSIFIERS[model_type]

    experiment = comet_ml.Experiment(
        project_name="baselines-v2",
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
            "model_type": model_type,
            "n_runs": n_runs,
        }
    )

    for _ in trange(n_runs, desc="Running experiments"):
        with experiment.train():
            X, y = shuffle(X, y)  # type:ignore
            model = ModelClass(**kwargs)
            if "experiment" in inspect.signature(model.fit).parameters:
                model.fit(
                    X, y, experiment=experiment
                )  # TODO: Log the accuracy at each iteration instead of just the ending accuracy
            else:
                model.fit(X, y)

        with experiment.test():
            experiment.log_metric("accuracy", model.score(tX, ty))
            experiment.log_metric("iterations", model.n_iter_)
    experiment.end()


if __name__ == "__main__":
    comet_ml.login()

    N_RUNS = 100
    for dirpath in DATA_DIRS:
        for model_type in get_args(MODEL_TYPE):
            print(f"Training {model_type}")
            run_experiment(
                n_runs=N_RUNS,
                model_type=model_type,
                dirpath=dirpath,
            )
