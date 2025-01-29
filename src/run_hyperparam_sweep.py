"""Runs 10-fold cross-validation to optimize the training subset size parameter"""

from concurrent.futures import ProcessPoolExecutor
from typing import Literal, get_args

import numpy as np
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from experiment_utils import ARCH_TYPE, DATA_DIRS, _make_model
from implementation.loader import load_dir

# hyparparameter settings
TRAINING_SUBSET_SIZES = [(i + 1) / 10 for i in range(10)]
LEARNING_EPOCHS_SETTINGS = range(1, 6)

NUM_FOLDS = 3
NUM_LEARNERS = 100


def optimize_over_dataset(dirpath, ensemble_type, pos):
    dataset_name = dirpath.split("/")[-1]
    results = np.zeros(
        (
            len(TRAINING_SUBSET_SIZES),
            len(LEARNING_EPOCHS_SETTINGS),
            NUM_FOLDS,
            len(get_args(ARCH_TYPE)),
        )
    )
    pbar = tqdm(desc=dataset_name, total=results.size, position=pos)

    (X, y), (tX, ty) = load_dir(dirpath)

    # Split the training data into subsets
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
    for fold_index, splits in enumerate(kf.split(X, y)):
        train_split, dev_split = splits

        for subset_size_index, subset_size in enumerate(TRAINING_SUBSET_SIZES):
            for learning_epochs_index, learning_epochs_setting in enumerate(LEARNING_EPOCHS_SETTINGS):
                for i, arch in enumerate(get_args(ARCH_TYPE)):
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
                    # accuracy = sklearn.metrics.accuracy_score(y[dev_split], preds)
                    results[subset_size_index, learning_epochs_index, fold_index, i] = f1
                    pbar.update()

    results = np.mean(results, axis=(-2, -1))
    np.savetxt(
        f"results/{dataset_name}.{ensemble_type}.sweep.csv",
        results,
        delimiter=",",
        fmt="%.3e",
    )


if __name__ == "__main__":
    # Optimize each dataset individually
    with ProcessPoolExecutor() as executer:
        for i, dir in enumerate(DATA_DIRS):
            for j, ensemble_type in enumerate(get_args(Literal["voted", "soup"])):
                executer.submit(optimize_over_dataset, dir, ensemble_type, i + (j * len(DATA_DIRS)))
