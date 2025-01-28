from typing import Any, Type

import numpy as np
import sklearn.metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from .loader import resample_if

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
    def fit(self, X, y):
        num_labels = len(set(y))

        for _ in range(self.n_learners):
            learner = self.LearnerClass(**self.learner_kwargs)
            while True:
                X_subset, y_subset = resample_if(X, y, self.training_size or 1)  # type:ignore

                # Repeat if we chose a subset missing labels
                if len(set(y_subset)) == num_labels:
                    break
            learner.fit(X_subset, y_subset)
            self.clfs.append(learner)

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
        return sklearn.metrics.accuracy_score(y, preds)
