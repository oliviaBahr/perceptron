import inspect
from typing import Any, Literal, get_args

import comet_ml
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from tqdm import trange

import addmlpclassifier
import addsgdclassifier
from loader import Loader

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


def run_experiment(
    n_runs: int,
    model_type: MODEL_TYPE,
    dataset_name: str,
    train_path: str,
    test_path: str,
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
            "dataset": dataset_name,
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

    DATA_PATHS = [
        (
            "cod-rna",
            "./classification-data/cod-rna/train.txt",
            "./classification-data/cod-rna/test.txt",
        ),
        ("cov", "./classification-data/cov/covtype.libsvm.binary", None),
        ("german", "./classification-data/german/german.numer", None),
        (
            "ijcnn",
            "./classification-data/ijcnn/ijcnn1",
            "./classification-data/ijcnn/ijcnn1.t",
        ),
        (
            "imdb",
            "./classification-data/imdb/train_labeledBow.feat",
            "./classification-data/imdb/test_labeledBow.feat",
        ),
        ("phishing", "./classification-data/phishing/phishing.txt", None),
        (
            "poker",
            "./classification-data/poker/poker",
            "./classification-data/poker/poker.t",
        ),
        ("real-sim", "./classification-data/real-sim/real-sim", None),
        ("skin", "./classification-data/skin/skin_noskin.txt", None),
        ("webspam", "./classification-data/webspam/unigram.svm", None),
    ]

    N_RUNS = 100
    for dataset_name, train_path, test_path in DATA_PATHS:
        for model_type in get_args(MODEL_TYPE):
            print(f"Training {model_type}")
            run_experiment(
                n_runs=N_RUNS,
                model_type=model_type,
                dataset_name=dataset_name,
                train_path=train_path,
                test_path=test_path,
            )
