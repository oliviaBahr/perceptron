"""Runs experiments across a grid of two hyperparameters, creating a heatmap of their results"""

from itertools import product
import multiprocessing
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from loader import Loader
from run_baseline_experiments import CLASSIFIERS, MODEL_TYPE
from typing import Collection, Iterable, Tuple
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

original_data = None
ModelClass = None
kwarg_labels = None

def _score_run(a_b: Tuple[int, int]):
    """Score a single run.

    Args:
        a_b (Tuple[int, int]): A tuple of the two hyperparameter values
    """
    global original_data, ModelClass, kwarg_labels

    if original_data is None or ModelClass is None or kwarg_labels is None:
        raise Exception()

    (X, y), (tX, ty) = original_data
    X, y = shuffle(X, y)
    a, b = a_b
    model = ModelClass(**{kwarg_labels[0]: a, kwarg_labels[1]: b})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X, y, None)
    return model.score(tX, ty)

def run_hyperparam_search(
    model_type: MODEL_TYPE,
    kwarg_a: Tuple[str, Collection],
    kwarg_b: Tuple[str, Collection],
):
    """Runs all combinations over the two hyperparameter ranges and creates a heatmap.

    Args:
        model_type (MODEL_TYPE): The class of model
        kwarg_a (Tuple[str, range]): A tuple of (kwarg_name, kwarg_range)
        kwarg_b (Tuple[str, range]): A tuple of (kwarg_name, kwarg_range)
    """
    global original_data, ModelClass, kwarg_labels

    # Put shared static variables in global scope
    #  so they will be shared with copy-on-write across processes
    original_data = Loader.load("./data/imdb/train_labeledBow.feat", "./data/imdb/test_labeledBow.feat")
    ModelClass, kwargs = CLASSIFIERS[model_type]
    kwarg_labels = (kwarg_a[0], kwarg_b[0])

    # Run experiments in parallel
    total_runs = len(kwarg_a[1]) * len(kwarg_b[1])
    metric_values = process_map(_score_run, product(kwarg_a[1], kwarg_b[1]), total=total_runs)
    metric_values = np.array(metric_values).reshape((len(kwarg_a[1]), len(kwarg_b[1])))

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(metric_values.T)
    ax.set(xlabel=kwarg_a[0], ylabel=kwarg_b[0])
    plt.gca().invert_yaxis()
    plt.xticks(ticks=range(metric_values.shape[0]), labels=[str(a) for a in kwarg_a[1]])
    plt.yticks(ticks=range(metric_values.shape[1]), labels=[str(b) for b in kwarg_b[1]])
    plt.show()

if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    run_hyperparam_search(
        model_type='add_perc',
        kwarg_a=("num_learners", range(1, 101)),
        kwarg_b=("epoch_size", [x / 100 for x in range(10, 110, 10)]),
    )
