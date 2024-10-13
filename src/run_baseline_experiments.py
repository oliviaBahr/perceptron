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
        # run = neptune.init_run(project="adding-perceptron/baseline-comparison", dependencies="infer")
        # run["params"] = {
        #     "dataset": dataset_name,
        #     "model_type": model_type,
        # }
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

        X, y = shuffle(X, y)
        model = ModelClass(**kwargs)
        model.fit(X, y) # TODO: Log the accuracy at each iteration instead of just the ending accuracy

        experiment.test()
        experiment.log_metric("accuracy", model.score(tX, ty))
        experiment.log_metric("iterations", model.n_iter_)

        # fig = variance_plots(run, classifiers, data_name)
        # run["variance_plot"].upload(fig)
        experiment.end()


# def variance_plots(run, classifiers: dict, data_name) -> plt.Figure:
#     run.wait()  # wait for neptune data to sync

#     scores = [run[f"{clf_name}/acc"].fetch_values().value.tolist() for clf_name in classifiers]
#     scores = np.array(scores).T

#     iter_counts = [run[f"{clf_name}/iters"].fetch_values().value.tolist() for clf_name in classifiers]
#     iter_counts = np.array(iter_counts).T

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
#     fig.suptitle(data_name)

#     ax1.violinplot(scores, showmeans=True)
#     ax1.set_xticks(np.arange(1, len(classifiers) + 1))
#     ax1.set_xticklabels(classifiers.keys())
#     ax1.set_ylabel("Accuracy")

#     ax2.set_xticks(np.arange(1, len(classifiers) + 1))
#     ax2.set_xticklabels(classifiers.keys())
#     ax2.set_ylabel("Iterations")
#     parts = ax2.violinplot(iter_counts, showmeans=True)
#     _ = [pc.set_facecolor("red") for pc in parts["bodies"]]
#     _ = [parts[bar].set_color("red") for bar in ["cmaxes", "cmins", "cbars", "cmeans"]]

#     fig.tight_layout()
#     return fig


if __name__ == "__main__":
    comet_ml.login()

    DATA_PATHS = [
        ("imdb", "./data/imdb/train_labeledBow.feat", "./data/imdb/test_labeledBow.feat"),
        # ("cod-rna", "../data/cod-rna/train.csv", "../data/cod-rna/test.csv"),
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
                test_path=test_path
            )
