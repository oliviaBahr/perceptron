import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample

from src.loader import Loader


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

        self.n_learners = kwargs.get("max_iter", 100)
        kwargs.pop("random_state", None)
        kwargs["max_iter"] = 1
        self.epoch_size = kwargs.pop("epoch_size", 1.0)
        self.kwargs = kwargs  # Store remaining kwargs for reuse in fit()
        self.clf = SGDClassifier(**kwargs)
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
        learner = SGDClassifier(**self.kwargs, random_state=random.randint(0, 100000000))
        learner.fit(*Loader.resample_if(X, y, self.epoch_size))
        return learner

    def fit(self, X, y):
        # setup and initial scores
        (X, y), (dX, dy) = Loader.dev_split(X, y)
        self.clf.fit(*Loader.resample_if(X, y, self.epoch_size))
        self.dev_scores.append(self.clf.score(dX, dy))
        self.scores.append(self.clf.score(X, y))

        # multiprocess learners
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._train_one_learner, X, y) for _ in range(self.n_learners - 1)
            ]

        for future in as_completed(futures):
            # sum weights
            learner = future.result()
            self.clf.coef_ += learner.coef_
            self.clf.intercept_ += learner.intercept_

            # record
            self.dev_scores.append(self.clf.score(dX, dy))
            self.scores.append(self.clf.score(X, y))
            self.n_iter_ += 1

            # early stop
            if self._early_stop():
                for future in futures:
                    future.cancel()
                break

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
