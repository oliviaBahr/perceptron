import unittest
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from addmlpclassifier import AddMLPClassifier
from addsgdclassifier import AddSGDClassifier
from loader import Loader

warnings.filterwarnings("ignore")


class TestClassifiers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        # Generate synthetic data once for all tests
        cls.X, cls.y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, n_clusters_per_class=2, random_state=42
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def check_classifier(self, clf, X_train, y_train, X_test, y_test):
        # Basic checks
        self.assertTrue(hasattr(clf, "clf"), "Missing clf attribute")
        self.assertTrue(hasattr(clf, "scores"), "Missing scores attribute")
        self.assertTrue(hasattr(clf, "dev_scores"), "Missing dev_scores attribute")

        # Check predictions
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(f"Train score: {train_score:.3f}")
        print(f"Test score: {test_score:.3f}")

        # Check predictions shape
        preds = clf.predict(X_test)
        self.assertEqual(len(preds), len(y_test), "Prediction length mismatch")

    def test_sgd_perceptron(self):
        config = {
            "loss": "perceptron",
            "learning_rate": "constant",
            "eta0": 1,
            "penalty": None,
            "max_iter": 2,
        }
        print(f"\nSGD (perceptron): {config}")
        clf = AddSGDClassifier(**config)
        clf.fit(self.X_train, self.y_train)
        self.check_classifier(clf, self.X_train, self.y_train, self.X_test, self.y_test)

    def test_sgd_log_loss(self):
        config = {"loss": "log_loss", "penalty": None, "max_iter": 2}
        print(f"\nSGD (log loss): {config}")
        clf = AddSGDClassifier(**config)
        clf.fit(self.X_train, self.y_train)
        self.check_classifier(clf, self.X_train, self.y_train, self.X_test, self.y_test)

    def test_sgd_hinge(self):
        config = {"loss": "hinge", "penalty": None, "max_iter": 2}
        print(f"\nSGD (hinge): {config}")
        clf = AddSGDClassifier(**config)
        clf.fit(self.X_train, self.y_train)
        self.check_classifier(clf, self.X_train, self.y_train, self.X_test, self.y_test)

    def test_mlp(self):
        config = {"max_iter": 2, "epoch_size": 0.5}
        print(f"\nMLP: {config}")
        clf = AddMLPClassifier(**config)
        clf.fit(self.X_train, self.y_train)
        self.check_classifier(clf, self.X_train, self.y_train, self.X_test, self.y_test)

    def test_imdb(self):
        clf = AddSGDClassifier(loss="perceptron", learning_rate="constant", eta0=1, penalty=None, max_iter=2)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = Loader.load(
            "./classification-data/imdb/train_labeledBow.feat", "./classification-data/imdb/test_labeledBow.feat"
        )
        clf.fit(self.X_train, self.y_train)
        self.check_classifier(clf, self.X_train, self.y_train, self.X_test, self.y_test)


if __name__ == "__main__":
    unittest.main()
