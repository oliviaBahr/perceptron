import warnings, csv, os
from contextlib import contextmanager
import matplotlib.pyplot as plt
from typing import Literal, Iterator, Generator, Annotated
from pydantic import validate_call, Field
from scipy.sparse import spmatrix
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import lru_cache
from itertools import cycle
from codetiming import Timer
from numpy import ndarray, array_split, unique
from sklearn.linear_model import Perceptron as skPerc
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Loader:
    @staticmethod
    def get_name(pathname) -> str:
        return pathname.split("/")[-2].lower()

    @staticmethod
    @lru_cache
    def load(path1, path2=None, test_size=None) -> tuple[spmatrix, ndarray]:
        if path2:
            return load_svmlight_file(path1), load_svmlight_file(path2)
        else:
            data = load_svmlight_file(path1)
            X, tX, y, ty = train_test_split(data[0], data[1], test_size=test_size)
            return (X, y), (tX, ty)

    @staticmethod
    @lru_cache
    def load_imdb_binary(trainpath, testpath) -> tuple[spmatrix, ndarray]:
        X, y, tX, ty = [], [], [], []

        # funcs to make features and classes binary
        featfunc = lambda x: 1 if x > 0 else 0
        classfunc = lambda x: 1 if x > 4 else 0

        for line in open(trainpath):
            line = line.strip().split()
            # classes
            y.append(classfunc(int(line.pop(0))))

            # features
            feats = [itm.split(":") for itm in line]
            X.append({int(i): featfunc(float(v)) for i, v in feats})

        for line in open(testpath):
            line = line.strip().split()
            # classes
            ty.append(classfunc(int(line.pop(0))))

            # features
            feats = [itm.split(":") for itm in line]
            tX.append({int(i): featfunc(float(v)) for i, v in feats})

        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(X)
        tX = vectorizer.transform(tX)
        return (X, y), (tX, y)


class Perceptron:
    def __init__(self, trainpath, testpath=None, test_size=None, dataset_name=None) -> None:
        # base perceptron to store weights
        self.base = skPerc(max_iter=1)

        # data
        self.dataset_name = dataset_name if dataset_name else Loader.get_name(trainpath)
        if "imdb" in self.dataset_name.lower():
            traindata, testdata = Loader.load_imdb_binary(trainpath, testpath)
        else:
            traindata, testdata = Loader.load(trainpath, testpath, test_size)
        self.X, self.y = traindata
        self.tX, self.ty = testdata

        # dataset info
        self.train_size = self.X.shape[0]
        self.test_size = self.tX.shape[0]
        self.n_classes = len(unique(self.y))
        self.n_features = self.X.shape[1]
        self.n_weights = 1 if self.n_classes == 2 else self.n_classes

        # training run info
        self.accuracies = None
        self.epoch_size = None
        self.ensemble_size = None
        self.data_opts = None
        self.train_time = None

        # logging and writing options
        self.outfile = None
        self.write = None
        self.log = None

    # wraps the training
    @contextmanager
    def _training_context(self, train_opts: dict):
        # set opts as object attributes ("data_opts", "ensemble_size", etc.)
        del train_opts["self"]
        for key, value in train_opts.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # reset base perceptron and accuracies
        self.base = skPerc(max_iter=1)
        self.accuracies = []

        # log
        if self.log != "none":
            print(f"Training on {self.dataset_name}...")

        # start timer
        timer = Timer(logger=None)
        timer.start()

        try:
            yield  # the code inside the with block

        finally:
            # stop timer
            dur = timer.stop()
            self.train_time = float(f"{dur:.2f}")

            # write to csv
            if self.write:
                self._write_results(self.outfile)

            # log info
            if self.log != "none":
                print(f"Training time: {self.train_time_str()}")
                self.print_info()

    # TRAINING
    @validate_call
    def train(
        self,
        ensemble_size: Annotated[int, Field(gt=0)] = 50,
        epoch_size: Annotated[float, Field(gt=0, le=1)] = 1.0,
        data_opts: Literal["partial", "cycle", "window", "whole"] = "whole",
        log: Literal["all", "min", "none"] = "all",
        write: bool = True,
        outfile: str = "training_runs.csv",
    ) -> None:
        """
        Trains the perceptron model.

        Parameters:
            ensemble_size (int):
            The number of weak learners in the ensemble.

            epoch_size (float):
            The fraction of the training data to use in each epoch.

            data_opts (Literal["partial", "cycle", "window"]):
            The data sampling option.
            "whole": Uses the entire training dataset in each epoch.
            "partial": Uses a partial subset of the training data in each epoch.
            "cycle": Cycles through different subsets of the training data in each epoch.
            "window": Uses a sliding window approach to sample the training data in each epoch.

            log (Literal["all", "min", "none"]):
            The level of logging during training.
            "all": Log all information.
            "min": Log minimal information.
            "none": Do not log any information.

            write (bool):
            Whether to write the results to a file.

            outfile (str):
            The file to save the results to.

        Returns:
            None
        """

        with self._training_context(locals()):  # set training options and start timer
            data = self._data_generator()  # infinite data according to data_opts
            self.base.fit(*next(data))  # initial fit to set up base perceptron

            # loop over ensemble of perceptrons trained for one epoch each
            for weak_learner in self._train_ensemble_parallel(data):  # swap the training function for parallel or sequential
                self._add_weights(weak_learner)  # add weights of new weak learner
                self._test_and_record()  # get acc of updated ensemble
                if self.log == "all":
                    print(self.accuracies[-1])

    def _train_ensemble_parallel(self, data) -> Generator[skPerc, None, None]:
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(self.ensemble_size - 1):
                futures.append(executor.submit(skPerc(max_iter=1).fit, *next(data)))

            for future in as_completed(futures):
                yield future.result()

    def _train_ensemble_sequential(self, data) -> Generator[skPerc, None, None]:
        for _ in range(self.ensemble_size - 1):
            yield skPerc(max_iter=1).fit(*next(data))

    def _add_weights(self, new: skPerc) -> None:
        self.base.coef_ = self.base.coef_ + new.coef_
        self.base.intercept_ = self.base.intercept_ + new.intercept_

    def _data_generator(self) -> Iterator:
        X, y = shuffle(self.X, self.y)
        match self.data_opts:
            case "whole":
                return cycle([(X, y)])
            case "partial":
                split_size = round(self.epoch_size * self.train_size)
                return cycle([(X[:split_size], y[:split_size])])
            case "cycle":
                n_splits = round(1 / self.epoch_size)
                X_groups = array_split(X.A, n_splits)
                y_groups = array_split(y, n_splits)
                return cycle(zip(X_groups, y_groups))
            case "window":
                split_size = round(self.epoch_size * self.train_size)
                ratio = (self.train_size - split_size) / self.ensemble_size
                step = round(ratio) if ratio >= 1 else 1

                def tuple_window() -> Generator[tuple, None, None]:
                    for i in cycle(range(0, self.train_size, step)):
                        end = i + split_size
                        if end <= self.train_size:
                            Xwin = X[i:end]
                            ywin = y[i:end]
                        else:
                            Xwin = X[i:] + X[: end - self.train_size]
                            ywin = y[i:] + y[: end - self.train_size]
                        yield (Xwin, ywin)

                return tuple_window()
            case _:
                raise ValueError("data_opts must be 'partial', 'cycle', 'window', or 'whole")

    def _test_and_record(self) -> float:
        acc = self.base.score(self.tX, self.ty)
        self.accuracies.append(acc)

    def _write_results(self, outfile) -> None:
        with open(outfile, "a") as f:
            writer = csv.writer(f)
            # write header if file is empty
            if os.path.getsize(outfile) == 0:
                header = ["dataset", "max_acc", "last_acc", "ensemble_size", "epoch_size", "data_opts", "train_time", "all_accs"]
                writer.writerow(header)
            writer.writerow([self.dataset_name, max(self.accuracies), self.accuracies[-1], self.ensemble_size, self.epoch_size, self.data_opts, self.train_time, self.accuracies])

    def print_info(self, include: list[str] = None) -> None:
        data = {
            "DATASET": [
                ("Dataset", self.dataset_name),
                ("Train size", self.train_size),
                ("Test size", self.test_size),
                ("Classes", self.n_classes),
                ("Features", self.n_features),
            ],
            "RUN INFO": [
                ("Ensemble", self.ensemble_size),
                ("Epoch size", self.epoch_size),
                ("Data opts", self.data_opts),
                ("Train time", self.train_time_str()),
                ("Max at", self.accuracies.index(max(self.accuracies)) + 1),
                ("Max acc", f"{max(self.accuracies):.4f}"),
                ("Final acc", f"{self.accuracies[-1]:.4f}"),
            ],
        }
        # loop over data and print
        for section, info in data.items():
            if not include or section.upper() in include:
                print(f"\n{section:-^20}")
                for k, v in info:
                    print(f"{k+':':<12}{v}")

    def train_time_str(self) -> str:
        if self.train_time < 0:
            return "Incomplete"
        elif self.train_time < 60:
            return f"{self.train_time}s"
        else:
            hours, rem = divmod(self.train_time, 3600)
            mins, secs = divmod(rem, 60)
            hours = f"{hours}h " if hours > 0 else ""
            return f"{hours}{mins}m {secs}s"

    def plot(self):
        plt.plot(self.accuracies)
        plt.title(f"{self.dataset_name}")
        plt.xlabel("Ensemble Size")
        plt.ylabel("Accuracy")
        plt.ylim(0.5, 1)
        plt.show()


if __name__ == "__main__":
    ...
