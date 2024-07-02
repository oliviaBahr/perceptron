import warnings
import csv
import matplotlib.pyplot as plt
from typing import Tuple, Literal, Iterator
from scipy.sparse import spmatrix
from functools import lru_cache
from itertools import cycle
from more_itertools import windowed
from codetiming import Timer
from numpy import ndarray, array_split, unique, zeros, array
from sklearn.linear_model import Perceptron as skPerc
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Loader:
    @staticmethod
    def split(data, test_size=0.2) -> tuple[tuple, tuple]:
        X, tX, y, ty = train_test_split(data[0], data[1], test_size=test_size)
        return (X, y), (tX, ty)

    @staticmethod
    @lru_cache
    def load_svm(pathname) -> tuple[spmatrix, ndarray]:
        return load_svmlight_file(pathname)
    
    @staticmethod
    @lru_cache
    def load_imdb_binary(pathname):
        lines = [line.strip().split() for line in open(pathname)]
        X, y = [], []

        # funcs to make features and classes binary
        featfunc = lambda x: 1 if x > 0 else 0
        classfunc = lambda x: 1 if x > 4 else 0
        
        for line in lines:
            # classes
            y.append(classfunc(int(line.pop(0))))
            
            # features
            feats = [itm.split(":") for itm in line]
            X.append({int(i): featfunc(float(v)) for i, v in feats})
        
        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(X)
        return X, array(y)  


class Perceptron:
    def __init__(self, dataset_name: str, traindata: Tuple[spmatrix | ndarray], testdata: Tuple[spmatrix | ndarray], outfile="training_runs.csv") -> None:
        # base perceptron to track summed weights
        self.base = skPerc()
        self.timer = Timer()

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
        self.epoch_size = None
        self.ensemble_size = None
        self.data_opts = None
        self.train_time = None

    def reset(self) -> None:
        self.base = skPerc()
        self.accuracies = []

    def train(self, ensemble_size=50, epoch_size=1.0, data_opts: Literal["partial", "cycle", "window", "whole"]="whole", log=True, write=True, outfile="training_runs.csv") -> None:
        """
        Trains the perceptron model.

        ### Args:
            `ensemble_size (int, optional)`:
                The number of weak learners in the ensemble. Defaults to `50`.

            `epoch_size (float, optional)`:
                The fraction of the training data to use in each epoch. Defaults to `1.0`.

            `data_opts (Literal["partial", "cycle", "window"], optional)`:
                The data sampling option. Defaults to `"whole"`.
                `"whole"`: Uses the entire training dataset in each epoch.
                `"partial"`: Uses a partial subset of the training data in each epoch.
                `"cycle"`: Cycles through different subsets of the training data in each epoch.
                `"window"`: Uses a sliding window approach to sample the training data in each epoch.

            `log (bool, optional)`:
                Whether to log the accuracy during training. Defaults to `True`.
            
            `write (bool, optional)`:
                Whether to write the results to a file. Defaults to `True`.
            
            `outfile (str, optional)`:
                The file to save the results to. Defaults to `"training_runs.csv"`.

        ### Returns:
            `None`
        """
        assert (epoch_size > 0), "epoch_size must be greater than 0"
        assert (ensemble_size > 0 and isinstance(ensemble_size, int)), "ensemble_size must be a positive integer"
        assert (data_opts in ["partial", "cycle", "window", "whole"]), "data_opts must be one of 'partial', 'cycle', 'window', or 'whole'"
        
        # record
        self.epoch_size = epoch_size
        self.ensemble_size = ensemble_size
        self.data_opts = data_opts
        self.train_time = -1  # incomplete

        ## TRAINING
        self.timer.start()
        self.reset()
        data = self.data_generator() # infinite data according to data_opts
        self.base.fit(*next(data)) # initial fit to set up base perceptron
        
        for _ in range(ensemble_size - 1):
            new_perc = skPerc(max_iter=1).fit(*next(data))
            self.add_weights(new_perc)
            self.test_and_record()
            if log: print(self.accuracies[-1])

        # record time and write to csv
        self.train_time = float(f"{self.timer.stop():.2f}")
        if write: self.write_results(outfile)

    def add_weights(self, new: skPerc) -> None:
        self.base.coef_ = self.base.coef_ + new.coef_
        self.base.intercept_ = self.base.intercept_ + new.intercept_    

    def data_generator(self) -> Iterator:
        X, y = shuffle(self.X, self.y)
        split_size = round(self.epoch_size*self.train_size)
        n_splits = round(1/self.epoch_size)
        match self.data_opts:
            case "whole":
                return cycle([(X, y)])
            case "partial":
                return cycle([(X[:split_size], y[:split_size])])
            case "cycle":
                X_groups = array_split(X, n_splits)
                y_groups = array_split(y, n_splits)
                return cycle(zip(X_groups, y_groups))
            case "window":
                ratio = (self.train_size-split_size) / self.ensemble_size
                data = zip(cycle(X), cycle(y))
                return windowed(data, n=split_size, step=(1 if ratio<1 else int(ratio)))
    
    def test_and_record(self) -> float:
        acc = accuracy_score(self.ty, self.base.predict(self.tX))
        self.accuracies.append(acc)

    def write_results(self, outfile) -> None:
        with open(outfile, "a") as f:
            # dataset, max_acc, last_acc, ensemble_size, epoch_size, data_opts, train_time, all_accs
            writer = csv.writer(f)
            writer.writerow([self.dataset_name, max(self.accuracies), self.accuracies[-1], self.ensemble_size, self.epoch_size, self.data_opts, self.train_time, self.accuracies])

    def plot(self):
        plt.plot(self.accuracies)
        plt.title(f"{self.dataset_name}")
        plt.xlabel("Ensemble Size")
        plt.ylabel("Accuracy")
        plt.ylim(.5, 1)
        plt.show()


if __name__ == "__main__":
    ... 