from comet_ml import Experiment
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


class AddMLPClassifier:

    def __init__(self, **kwargs):
        """Olivia's additive maneuver implemented for sklearn's MLPClassifier."""
        self.hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100,))
        self.max_iter = kwargs.get("max_iter", 100)  # number of learners
        random.seed(a=kwargs.get("random_state", 42))
        kwargs["max_iter"] = 1
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.clf = MLPClassifier(**kwargs)
        self.best = self.clf
        self.scores = []
        self.n_iter_ = 0

    def _shuffler(self, X, y):
        """Resample part of data if epoch_size < 1.0."""
        if self.epoch_size < 1.0:
            return resample(X, y, replace=False, n_samples=int(X.shape[0] * self.epoch_size))
        return X, y

    def _early_stop(self):
        if len(self.scores) > 1 and self.scores[-1] > max(self.scores[:-1]):
            self.best = self.clf

        if self.scores.index(max(self.scores)) < self.n_iter_ - 25:
            self.clf = self.best
            return True
        return False

    def _train_one_learner(self, X, y):
        learner = MLPClassifier(**self.kwargs, max_iter=1, random_state=random.randint(0, 1000000))
        learner.fit(*self._shuffler(X, y))
        return learner

    def fit(self, X, y, experiment: Experiment | None = None):
        self.clf.fit(*self._shuffler(X, y))
        self.scores.append(self.clf.score(X, y))
        self.n_iter_ = 1

        # multiprocessing learners
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._train_one_learner, X, y) for _ in range(self.max_iter - 1)]

        # average weights
        for future in as_completed(futures):
            learner = future.result()
            iters = self.n_iter_

            for i, (prev_weights, new_weights) in enumerate(zip(self.clf.coefs_, learner.coefs_)):
                self.clf.coefs_[i] = (prev_weights * iters + new_weights) / (iters + 1)

            for i, (prev_weights, new_weights) in enumerate(zip(self.clf.intercepts_, learner.intercepts_)):
                self.clf.intercepts_[i] = (prev_weights * iters + new_weights) / (iters + 1)

            # score
            self.n_iter_ += 1
            self.scores.append(self.clf.score(X, y))

            # comet logging
            if experiment is not None:
                experiment.log_metric("accuracy", value=self.scores[-1], step=self.n_iter_)

            if self._early_stop():
                for f in futures: # Cancel remaining futures
                    f.cancel()
                break
    
    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
