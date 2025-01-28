"""Runs 10-fold cross-validation to optimize the training subset size parameter"""

from concurrent.futures.process import ProcessPoolExecutor
from typing import Literal, get_args

import numpy as np
import sklearn.metrics
from sklearn.model_selection import KFold
from tqdm import tqdm

from experiment_utils import ARCH_TYPE, DATA_DIRS, _make_model
from implementation.loader import load_dir

# hyparparameter settings
TRAINING_SUBSET_SIZES = [(i + 1) / 10 for i in range(10)]
LEARNING_EPOCHS_SETTINGS = range(1, 6)

NUM_FOLDS = 10
NUM_LEARNERS = 50


def optimize_over_dataset(dirpath, pos):
    dataset_name = dirpath.split("/")[-1]
    results = np.zeros(
        (
            len(TRAINING_SUBSET_SIZES),
            len(LEARNING_EPOCHS_SETTINGS),
            NUM_FOLDS,
            len(get_args(ARCH_TYPE)),
            2,  # ensemble types
        )
    )
    pbar = tqdm(desc=dataset_name, total=results.size, position=pos)

    (X, y), (tX, ty) = load_dir(dirpath)

    # Split the training data into 10 subsets
    kf = KFold(n_splits=NUM_FOLDS)
    for fold_index, splits in enumerate(kf.split(X)):
        train_split, dev_split = splits

        for subset_size_index, subset_size in enumerate(TRAINING_SUBSET_SIZES):
            for learning_epochs_index, learning_epochs_setting in enumerate(LEARNING_EPOCHS_SETTINGS):
                for i, arch in enumerate(get_args(ARCH_TYPE)):
                    for j, ensemble_type in enumerate(get_args(Literal["voted", "soup"])):
                        model = _make_model(
                            arch=arch,
                            ensemble_type=ensemble_type,
                            max_epochs_per_learner=learning_epochs_setting,
                            num_learners=NUM_LEARNERS,
                            training_size=subset_size,
                            learner_kwargs={},
                        )
                        model.fit(X[train_split], y[train_split])  # type:ignore
                        preds = model.predict(X[dev_split])  # type:ignore
                        f1 = sklearn.metrics.f1_score(y[dev_split], preds, average="macro")
                        results[subset_size_index, learning_epochs_index, fold_index, i, j] = f1
                        pbar.update()

    breakpoint()
    results = np.mean(results, axis=(-3, -2, -1))
    breakpoint()
    np.savetxt(f"results/{dataset_name}.sweep.csv", results, delimiter=",", fmt="%.3e")


if __name__ == "__main__":
    results = []
    # Optimize each dataset individually
    with ProcessPoolExecutor() as executer:
        for i, dir in enumerate(DATA_DIRS):
            executer.submit(optimize_over_dataset, dir, i)
