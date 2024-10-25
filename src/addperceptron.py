from sklearn.linear_model import Perceptron
from sklearn.utils import resample
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

class AddPerceptron:

    def __init__(self, **kwargs):
        """Olivia's additive Perceptron implemented through sklearn."""
        self.max_iter = kwargs.get("max_iter", 100)  # Store passed max_iter since we rig it to 1
        random.seed(a=kwargs.get("random_state", 42))
        kwargs["max_iter"] = 1
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.clf = Perceptron(**kwargs)
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
        learner = Perceptron(max_iter=1)
        learner.fit(*self._shuffler(X, y))
        return learner

    def fit(self, X, y):
        self.clf.fit(*self._shuffler(X, y))
        self.scores.append(self.clf.score(X, y))
        self.n_iter_ = 1

        # multiprocessing learners
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._train_one_learner, X, y) for _ in range(self.max_iter - 1)]
        
        # sum weights
        for future in as_completed(futures):
            learner = future.result()
            self.clf.coef_ += learner.coef_
            self.clf.intercept_ += learner.intercept_
            self.scores.append(self.clf.score(X, y))
            self.n_iter_ += 1
            if self._early_stop():
                for future in futures:
                    future.cancel()
                break

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
