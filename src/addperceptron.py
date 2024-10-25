from sklearn.linear_model import Perceptron
from sklearn.utils import resample
import random


class AddPerceptron:

    def __init__(self, **kwargs):
        """Olivia's additive Perceptron implemented through sklearn."""
        self.max_iter = kwargs.get("max_iter", 1) # number of iterations per learner
        self.num_learners = kwargs.pop("num_learners", 100) # number of (weak) learners

        random.seed(a=kwargs.get("random_state", 42))
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.clf = Perceptron(**kwargs)
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

    def fit(self, X, y, *args, **kwargs):
        self.clf.fit(*self._shuffler(X, y))
        self.scores.append(self.clf.score(X, y))

        for _ in range(self.num_learners - 1):
            clfp = Perceptron(max_iter=self.max_iter, random_state=random.randint(0, 100000000))
            clfp.fit(*self._shuffler(X, y))
            self.clf.coef_ += clfp.coef_
            self.clf.intercept_ += clfp.intercept_
            self.scores.append(self.clf.score(X, y))
            self.n_iter_ += 1
            if self._early_stop():
                break

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
