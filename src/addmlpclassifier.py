import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from comet_ml import Experiment
from sklearn.neural_network import MLPClassifier

from src.loader import Loader


class AddMLPClassifier:

    def __init__(self, **kwargs):
        """Olivia's additive maneuver implemented for sklearn's MLPClassifier."""
        self.hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100,))
        self.max_iter = kwargs.get("max_iter", 100)  # number of learners
        random.seed(a=kwargs.get("random_state", 42))
        kwargs["max_iter"] = 1
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.kwargs = kwargs
        self.clf = MLPClassifier(**kwargs)
        self.best = self.clf
        self.scores = []
        self.dev_scores = []
        self.n_iter_ = 0

    def _early_stop(self):
        if len(self.dev_scores) > 1 and self.dev_scores[-1] > max(self.dev_scores[:-1]):
            self.best = self.clf

        if self.dev_scores.index(max(self.dev_scores)) < self.n_iter_ - 25:
            self.clf = self.best
            return True
        return False

    def _train_one_learner(self, X, y):
        learner = MLPClassifier(**self.kwargs, random_state=random.randint(0, 1000000))
        learner.fit(*Loader.resample_if(X, y, self.epoch_size))  # Use Loader.resample_if
        return learner

    def fit(self, X, y, experiment: Experiment | None = None):
        # Split data into training and development sets
        (X, y), (dX, dy) = Loader.dev_split(X, y)
        self.clf.fit(*Loader.resample_if(X, y, self.epoch_size))  # Use Loader.resample_if
        self.dev_scores.append(self.clf.score(dX, dy))  # Record dev score
        self.scores.append(self.clf.score(X, y))
        self.n_iter_ = 1

        # multiprocessing learners
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._train_one_learner, X, y) for _ in range(self.max_iter - 1)
            ]

        # average weights
        for future in as_completed(futures):
            learner = future.result()

            # sum weights
            for i, (prev_weights, new_weights) in enumerate(zip(self.clf.coefs_, learner.coefs_)):
                self.clf.coefs_[i] = (prev_weights * self.n_iter_ + new_weights) / (
                    self.n_iter_ + 1
                )

            for i, (prev_weights, new_weights) in enumerate(
                zip(self.clf.intercepts_, learner.intercepts_)
            ):
                self.clf.intercepts_[i] = (prev_weights * self.n_iter_ + new_weights) / (
                    self.n_iter_ + 1
                )

            # score
            self.n_iter_ += 1
            self.dev_scores.append(self.clf.score(dX, dy))  # Record dev score
            self.scores.append(self.clf.score(X, y))

            # comet logging
            if experiment is not None:
                experiment.log_metric("accuracy", value=self.scores[-1], step=self.n_iter_)

            if self._early_stop():
                for f in futures:  # Cancel remaining futures
                    f.cancel()
                break

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
