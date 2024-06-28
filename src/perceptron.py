from typing import Tuple
from itertools import cycle
from codetiming import Timer
from numpy import ndarray, array_split, unique, zeros
from scipy.sparse import spmatrix
from sklearn.linear_model import Perceptron as skPerc
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class Perceptron:
    def __init__(self, dataset_name: str, traindata: Tuple[spmatrix | ndarray], testdata: Tuple[spmatrix | ndarray]) -> None:
        # data
        self.X, self.y = traindata
        self.tX, self.ty = testdata

        # dataset info
        self.dataset_name = dataset_name
        self.train_size = self.X.shape[0]
        self.test_size = self.tX.shape[0]
        self.n_classes = len(unique(self.y))
        self.n_features = self.X.shape[1]
        self.n_weights = 1 if self.n_classes == 2 else self.n_classes

        # training run info
        self.accuracies = []
        self.epochs = None
        self.cycle = None
        self.ensemble_size = None
        self.train_time = None

    def test(self, clf) -> float:
        return accuracy_score(self.ty, clf.predict(self.tX))

    def train(self, ensemble_size=50, epochs=1, cycle_groups=None, log=True) -> None:
        """
        Trains the perceptron model.

        Args:
            `ensemble_size (int, optional)`: The number of perceptrons (weak learners) in the ensemble.
            `epochs (int, optional)`: The number of times each learner iteratesover the training data. < 1 trains on only part of the data.
            `cycle_groups (bool, optional)`: (if epochs < 1) Whether to cycle through the data groups or only use the first group.
            `log (bool, optional)`: Print accuracies during training.

        Raises:
            AssertionError: epochs > 0
            AssertionError: ensemble_size > 0 and isinstance(ensemble_size, int)

        Returns:
            None
        """
        
        assert (epochs > 0), "epochs must be greater than 0"
        assert (ensemble_size > 0 and isinstance(ensemble_size, int)), "ensemble_size must be a positive integer"
        
        # scikit shuffle
        X, y = shuffle(self.X, self.y)
        
        # split data for epochs < 1
        if epochs >= 1:
            data = zip([X], [y])
        else:
            n_splits = round(1/epochs)  # percent to number of groups
            X_groups = array_split(X, n_splits)
            y_groups = array_split(y, n_splits)

            if cycle_groups:
                self.cycle = True
                data = zip(X_groups, y_groups)
            else: # only use first group
                self.cycle = False
                data = zip([X_groups[0]], [y_groups[0]])
                
        ## TRAINING
        with Timer(initial_text="Training..."):
            # init base to keep track of coef and intercept
            base = skPerc()
            base.coef_ = zeros(self.n_weights, self.n_features)
            base.intercept_ = zeros(self.n_weights)
            
            for i, (X, y) in zip(range(ensemble_size), cycle(data)):
                # train a perceptron once over the data
                clf = skPerc(max_iter=1).fit(X, y)
                coef, intercept = clf.coef_, clf.intercept_

                # update base weights
                base.coef_ = base.coef_ + coef
                base.intercept_ = base.intercept_ + intercept
                
                # test
                self.accuracies.append(self.test(base))
                if log: print(i, ":", self.accuracies[-1])

        # record
        self.epochs = epochs
        self.ensemble_size = ensemble_size
        self.train_time = Timer().last


if __name__ == "__main__":
    ... 