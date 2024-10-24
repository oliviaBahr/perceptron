from sys import setdlopenflags
from comet_ml import Experiment
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import random


class AddMLPClassifier:

    def __init__(self, **kwargs):
        """Olivia's additive maneuver implemented for sklearn's MLPClassifier."""
        self.hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (100,))
        self.max_iter = kwargs.get("max_iter", 100)  # Store passed max_iter since we rig it to 1
        random.seed(a=kwargs.get("random_state", 42))
        kwargs["max_iter"] = 1
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.clf = MLPClassifier(**kwargs)
        self.best = self.clf
        self.scores = []
        self.n_iter_ = 1

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

    def fit(self, X, y, experiment: Experiment | None = None):
        self.clf.fit(*self._shuffler(X, y))
        self.scores.append(self.clf.score(X, y))
        (summed_coefs, summed_intercepts) = self.clf.coefs_, self.clf.intercepts_

        for learner_index in range(self.max_iter - 1):
            clfp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=1, random_state=random.randint(0, 100000000))
            clfp.fit(*self._shuffler(X, y))

            # Average weights
            summed_coefs = [prev_weights + new_weights for prev_weights, new_weights in zip(summed_coefs, clfp.coefs_)]
            summed_intercepts = [prev_weights + new_weights for prev_weights, new_weights in zip(summed_intercepts, clfp.intercepts_)]
            self.clf.coefs_ = [w / (learner_index + 2) for w in summed_coefs]
            self.clf.intercepts_ = [w / (learner_index + 2) for w in summed_intercepts]

            # Rescore method
            score = self.clf.score(X, y)
            self.scores.append(score)
            if experiment is not None:
                experiment.log_metric("accuracy", value=score, step=learner_index)
            self.n_iter_ += 1

            if self._early_stop():
                break

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
