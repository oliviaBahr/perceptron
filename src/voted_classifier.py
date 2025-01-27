import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Type

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from loader import dev_split, resample_if

ignore_warnings(category=ConvergenceWarning)  # type: ignore


class BaseLearner:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class VotedClassifier:
    def __init__(
        self,
        n_learners: int,
        training_size: float | None,
        LearnerClass: Type[BaseLearner],
        learner_kwargs: dict[str, Any],
    ):
        """Ensembled classifier, where inference is performed through majority voting

        Args:
            n_learners (int): The number of weak learners to train
            training_size (float | None): The size of the subset of data to use for training each learner. If `None`, use the entire training set.
            LearnerClass (Type[BaseLearner]): The class of individual weak learners
            learner_kwargs (dict[str, Any]): Initialization kwargs for weak learners
        """
        self.n_learners = n_learners
        self.training_size = training_size
        self.LearnerClass = LearnerClass
        self.learner_kwargs = learner_kwargs
        self.clfs: list[BaseLearner] = []

    @ignore_warnings(category=ConvergenceWarning)  # type: ignore
    def _train_one_learner(self, X, y):
        learner = self.LearnerClass(**self.learner_kwargs, random_state=random.randint(0, 100000000))
        learner.fit(X, y)
        return learner

    @ignore_warnings(category=ConvergenceWarning)  # type: ignore
    def fit(self, X, y):
        (X, y), (dX, dy) = dev_split(X, y)

        # Train learners in parallel
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._train_one_learner, *resample_if(X, y, self.training_size or 1))
                for _ in range(self.n_learners)
            ]

        dev_scores, scores = [], []
        for future in as_completed(futures):
            self.clfs.append(future.result())

            # record
            # dev_scores.append(self.score(dX, dy))
            # scores.append(self.score(X, y))

    def predict(self, X) -> np.ndarray:
        preds = np.array([clf.predict(X) for clf in self.clfs])  # (num_learners, |X|)
        u, indices = np.unique(preds, return_inverse=True)
        return u[
            np.argmax(
                np.apply_along_axis(
                    np.bincount,
                    0,
                    indices.reshape(preds.shape),
                    None,
                    np.max(indices) + 1,
                ),
                axis=0,
            )
        ]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        correct = np.sum(preds == y).item()
        return correct / y.shape[-1]
