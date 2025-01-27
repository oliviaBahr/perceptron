import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Type

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

    def score(self, X, y):
        pass


class SoupClassifier:
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
        self.souped_clf: BaseLearner | None = None

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
        count = 0
        for future in as_completed(futures):
            clf = future.result()
            if self.souped_clf is None:
                self.souped_clf = clf
            else:
                # Soup it up
                if hasattr(clf, "coef_"):
                    self.souped_clf.coef_ += clf.coef_  # type:ignore
                    self.souped_clf.intercept_ += clf.intercept_  # type:ignore
                elif hasattr(clf, "coefs_"):
                    old_coefs = self.souped_clf.coefs_  # type:ignore
                    new_coefs = clf.coefs_  # type:ignore
                    old_intercepts = self.souped_clf.intercepts_  # type:ignore
                    new_intercepts = clf.intercepts_  # type:ignore

                    self.souped_clf.coefs_ = [  # type:ignore
                        old + (new - old) / (count + 1) for old, new in zip(old_coefs, new_coefs)
                    ]
                    self.souped_clf.intercepts_ = [  # type:ignore
                        old + (new - old) / (count + 1) for old, new in zip(old_intercepts, new_intercepts)
                    ]
                else:
                    raise ValueError("Need coef_ or coefs_ to soup!")

            # record
            dev_scores.append(self.score(dX, dy))
            scores.append(self.score(X, y))
        return dev_scores, scores

    def predict(self, X):
        if self.souped_clf is None:
            raise ValueError()
        return self.souped_clf.predict(X)

    def score(self, *args, **kwargs):
        if self.souped_clf is None:
            raise ValueError()
        return self.souped_clf.score(*args, **kwargs)
