from typing import Any, Literal, get_args
import numpy as np
import matplotlib.pyplot as plt
import comet_ml
from sklearn.utils import shuffle
import addperceptron, addmlpclassifier, addsgdclassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from loader import Loader
from tqdm import trange


MODEL_TYPE = Literal['perc', 'add_perc', 'svm', 'add_svm', 'lr', 'add_lr', 'mlp', 'add_mlp']
CLASSIFIERS: dict[MODEL_TYPE, tuple[Any, dict[str, str | None]]] = {
    "perc": (Perceptron, {}),
    "add_perc": (addperceptron.AddPerceptron, {}),
    "svm": (SGDClassifier, {"loss": "hinge", "penalty": None}),
    "add_svm": (addsgdclassifier.AddSGDClassifier, {"loss": "hinge", "penalty": None}),
    "lr": (SGDClassifier, {"loss": "log_loss", "penalty": None}),
    "add_lr": (addsgdclassifier.AddSGDClassifier, {"loss": "log_loss", "penalty": None}),
    "mlp": (MLPClassifier, {}),
    "add_mlp": (addmlpclassifier.AddMLPClassifier, {}),
}

def run_experiment(
    n_runs: int,
    model_type: MODEL_TYPE,
    dataset_name: str,
    train_path: str,
    test_path: str
):
    """Runs experiments with a given model and dataset

    Args:
        n_runs (int): The number of repeated runs to perform
        model_type (Literal['perc', 'add_perc', 'svm', 'add_svm', 'lr', 'add_lr', 'mlp', 'add_mlp']): The type of model to train
        dataset_name (str): The name of the dataset (for logging purposes)
        train_path (str): Path to the train file
        test_path (str): Path to the test data file
    """
    (X, y), (tX, ty) = Loader.load(train_path, test_path)
    ModelClass, kwargs = CLASSIFIERS[model_type]

    for _ in trange(n_runs, desc="Running experiments"):
        experiment = comet_ml.Experiment(
          project_name="baselines",
          workspace="perceptrons",
          log_code = False,
          log_graph = False,
          log_git_metadata = False,
          log_git_patch = False,
          log_env_details = False,

        )
        experiment.log_parameters({
            "dataset": dataset_name,
            "model_type": model_type,
        })

        with experiment.train():
            X, y = shuffle(X, y)
            model = ModelClass(**kwargs)
            model.fit(X, y, experiment) # TODO: Log the accuracy at each iteration instead of just the ending accuracy

        with experiment.test():
            experiment.log_metric("accuracy", model.score(tX, ty))
            experiment.log_metric("iterations", model.n_iter_)
        experiment.end()


if __name__ == "__main__":
    comet_ml.login()

    DATA_PATHS = [
        ("imdb", "./data/imdb/train_labeledBow.feat", "./data/imdb/test_labeledBow.feat"),
        # ("cod-rna", "../data/cod-rna/train.csv", "../data/cod-rna/test.csv"),
    ]

    N_RUNS = 100
    for dataset_name, train_path, test_path in DATA_PATHS:
        for model_type in ['add_mlp']: # get_args(MODEL_TYPE):
            print(f"Training {model_type}")
            run_experiment(
                n_runs=N_RUNS,
                model_type=model_type,
                dataset_name=dataset_name,
                train_path=train_path,
                test_path=test_path
            )
