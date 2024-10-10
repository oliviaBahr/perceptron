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

    @property
    def n_iter_(self):
        return self.clf.n_iter_

    def _shuffler(self, X, y):
        """Resample part of data if epoch_size < 1.0."""
        if self.epoch_size < 1.0:
            return resample(X, y, replace=False, n_samples=int(X.shape[0] * self.epoch_size))
        return X, y

    def fit(self, X, y):
        self.clf.fit(*self._shuffler(X, y))
        for i in range(self.max_iter - 1):
            clfp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=1, random_state=random.randint(0, 100000000))
            clfp.fit(*self._shuffler(X, y))
            self.clf.coefs_ += clfp.coefs_
            self.clf.intercepts_ += clfp.intercepts_

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
