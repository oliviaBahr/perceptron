from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample
import random


class AddSGDClassifier:

    def __init__(self, **kwargs):
        """Olivia's additive linear classifier implemented through sklearn.

        Trains a linear classifier, implicitly ensembled from several weak learners
        by simply adding weights from each weak learner. Since the individual weak
        learners are not trained until convergence, regularization should not be used.

        Perceptron:
        ADDSGDClassifier(loss = "perceptron", learning_rate="constant", eta0 = 1, penalty = None)

        Logistic regression:
        ADDSGDClassifier(loss = "log_loss", penalty = None)

        SVM:
        ADDSGDClassifier(loss = "hinge", penalty = None)
        """

        self.max_iter = kwargs.get("max_iter", 100)  # Store passed max_iter since we rig it to 1
        kwargs.pop("random_state", None)
        kwargs["max_iter"] = 1
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.kwargs = kwargs  # Store remaining kwargs for reuse in fit()
        self.clf = SGDClassifier(**kwargs)
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

    def fit(self, X, y):
        self.clf.fit(*self._shuffler(X, y))
        self.scores.append(self.clf.score(X, y))

        for _ in range(self.max_iter - 1):
            clfp = SGDClassifier(random_state=random.randint(0, 100000000), **self.kwargs)
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
