import warnings, csv, os
from copy import deepcopy
from contextlib import contextmanager
from typing import Literal, Iterator, Generator, Annotated
from pydantic import validate_call, Field
from scipy.sparse import spmatrix, csr_matrix, vstack, hstack
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import lru_cache
from itertools import cycle
from numpy import ndarray, unique, array_equal
from sklearn.linear_model import Perceptron as skPerc
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from codetiming import Timer
from yaspin import yaspin
from yaspin.spinners import Spinners

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Loader:
    @staticmethod
    def get_name(pathname) -> str:
        return pathname.split("/")[-2].lower()

    @staticmethod
    @lru_cache
    def load(trainpath, testpath=None, test_size=None) -> tuple[spmatrix, ndarray]:
        if "imdb" in trainpath.lower():
            return Loader.load_imdb_binary(trainpath, testpath)
        elif not testpath:
            return Loader.load_split(trainpath, test_size)
        else:
            return load_svmlight_file(trainpath), load_svmlight_file(testpath)

    @staticmethod
    @lru_cache
    def load_split(trainpath, test_size=None) -> tuple[spmatrix, ndarray]:
        data = load_svmlight_file(trainpath)
        X, tX, y, ty = train_test_split(data[0], data[1], test_size=test_size, random_state=0)
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
        """
        Initializes the Perceptron model.
        Provide one path for all the data or two paths for separate training and test data.

        ### Parameters

        trainpath (str):
            The path to the training data (or all the data if there is no testdata).

        testpath (str):
            The path to the test data. Optional if all the data is in one file.

        test_size (float):
            The fraction of the data to use as test data if there is no testpath. 0.2 by default.

        dataset_name (str):
            The name of the dataset. If not provided, it is inferred from the trainpath.
        """
        # base perceptron to store weights
        self.base = None

        # data
        traindata, testdata = Loader.load(trainpath, testpath, test_size)
        self.X, self.devX, self.y, self.devy = train_test_split(*traindata, test_size=0.1)
        self.tX, self.ty = testdata

        # dataset info
        self.dataset_name = dataset_name if dataset_name else Loader.get_name(trainpath)
        self.train_size = self.X.shape[0]
        self.test_size = self.tX.shape[0]
        self.dev_size = self.devX.shape[0]
        self.n_classes = len(unique(self.y))
        self.n_features = self.X.shape[1]
        self.n_weights = 1 if self.n_classes == 2 else self.n_classes

        # training run info and trackers
        self.test_accs = None
        self.dev_accs = None
        self.epoch_size = None
        self.ensemble_size = None
        self.data_opts = None
        self.train_time = None
        self.stop_threshold = None
        self.pocket_best = None

        # logging and writing options
        self.outfile = None
        self.write = None
        self.log = None

    ## READ-ONLY PROPERTIES
    @property
    def accuracy(self, last_call: list[list] = ["x"]) -> float:
        assert self.base, "self.base is None. Cannot get accuracy from an untrained model."

        # use python's weird default arg behavior with mutable datatypes as a cache to avoid recalculating accuracy
        if last_call and array_equal(self.base.intercept_, last_call[0][0]):
            return last_call[0][1]

        acc = round(self.base.score(self.tX, self.ty), 7)  # test
        last_call[0] = [self.base.intercept_, acc]  # cache
        return acc

    @property
    def accuracy_string(self) -> str:
        return f"{self.accuracy:.2%}"

    @property
    def dev_accuracy(self) -> float:
        assert self.base, "self.base is None. Cannot get accuracy from an untrained model."
        return self.base.score(self.devX, self.devy)

    @property
    def n_iters_no_change(self) -> int:
        if not self.dev_accs:
            return 0
        return self.dev_accs[::-1].index(max(self.dev_accs))

    @property
    def n_iters(self) -> int:
        return len(self.dev_accs)

    @property
    def dev_max_at(self) -> int:
        return self.dev_accs.index(max(self.dev_accs)) + 1

    @property
    def test_max_at(self) -> int:
        return self.test_accs.index(max(self.test_accs)) + 1

    @property
    def train_time_string(self) -> str:
        if self.train_time < 0:
            return "Incomplete"
        elif self.train_time < 10:
            return f"{self.train_time:.2f}s"
        elif self.train_time < 60:
            return f"{round(self.train_time)}s"
        else:
            hours, rem = divmod(self.train_time, 3600)
            mins, secs = divmod(rem, 60)
            hours = f"{round(hours)}h " if hours > 0 else ""
            return f"{hours}{round(mins)}m {round(secs)}s"

    # wraps the training
    @contextmanager
    def _training_context(self, train_opts: dict):
        try:
            # reset base perceptron and accuracies
            self.base = skPerc(max_iter=1).fit(self.X, self.y)  # fit to get the right shapes for coef_ and intercept_
            self.base.coef_.fill(1)
            self.base.intercept_.fill(0)
            self.test_accs = []
            self.dev_accs = []

            # set opts as object attributes ("data_opts", "ensemble_size", etc.)
            del train_opts["self"]
            for key, value in train_opts.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # log
            if self.log != "none":
                print(f"\nTraining {self.ensemble_size} learners on {self.dataset_name} ({self.data_opts}, {self.epoch_size})...")

            # start spinner
            spinner = yaspin()
            if self.log not in ["all", "none"]:
                spinner.spinner = Spinners.balloon2
                spinner.start()

            # start timer
            timer = Timer(logger=None)
            timer.start()

            # the code inside the with block
            yield

        except Exception as e:
            self.train_time = -1
            timer.stop()
            spinner.stop()
            print(f"Training failed :(")
            raise  # re-raise the exception

        else:  # runs on success
            # stop timer and spinner
            dur = timer.stop()
            self.train_time = float(f"{dur:.2f}")
            spinner.stop()

            # write to csv
            if self.write:
                self._write_results(self.outfile)

            # log info
            if self.log != "none":
                print(f"Training time: {self.train_time_string}")
                if self.log == "min":
                    if self.n_iters < self.ensemble_size:
                        print(f"Stopped early at ensemble size {self.n_iters}")
                    print(f"Accuracy: {self.accuracy_string}\n")
                else:
                    self.print_info()
        finally:
            spinner.stop()

    # yields the data for training
    def _data_generator(self) -> Iterator:
        X, y = shuffle(self.X, self.y)
        match self.data_opts:
            case "whole":
                return cycle([(X, y)])

            case "partial":
                split_size = round(self.epoch_size * self.train_size)
                X_part, y_part = X[:split_size], y[:split_size]
                return cycle([(X_part, y_part)])

            case "cycle":
                if self.epoch_size > 0.5:
                    print("\nWarning: epoch_size > 0.5 for 'cycle' is equivalent to 'whole'.")
                    print("Setting epoch_size to 1.0 and data_opts to 'whole'")
                    self.epoch_size = 1.0
                    self.data_opts = "whole"

                split_size = round(max(1, self.epoch_size * self.train_size))
                groups = []

                for i in range(0, self.train_size, split_size):
                    end = i + split_size
                    if end <= self.train_size:
                        groups.append((X[i:end], y[i:end]))

                return cycle(groups)

            case "window":
                split_size = round(max(1, (self.epoch_size * self.train_size)))
                step = round(max(1, (self.train_size - split_size) / self.ensemble_size))
                windows = []

                for i in range(0, self.train_size, step):
                    end = i + split_size
                    if end <= self.train_size:
                        Xwin = X[i:end]
                        ywin = y[i:end]
                        windows.append((Xwin, ywin))
                    else:
                        Xwin = vstack([X[i:], X[: end - self.train_size]])
                        try:
                            ywin = y[i:] + y[: end - self.train_size]
                        except:
                            try:
                                ywin = hstack([y[i:], y[: end - self.train_size]]).T.toarray()
                            except:
                                ywin = hstack([y[i:], y[: end - self.train_size]])
                        windows.append((Xwin, ywin))

                return cycle(windows)

            case _:
                raise ValueError("data_opts must be 'partial', 'cycle', 'window', or 'whole'")

    # tracks best weights and stops training
    def _pocket_stop(self) -> bool:
        assert self.n_iters > 0, "self.n_iters must be > 0.\nNo training has been done, why are you trying to stop?"

        # track pocket_best weights
        if self.n_iters == 1 or self.dev_accs[-1] > max(self.dev_accs[:-1]):
            self.pocket_best = [self.base.coef_, self.base.intercept_]

        # early stopping
        reached_ensemble_size = lambda: self.n_iters >= self.ensemble_size
        reached_stop_threshold = lambda: not self.stop_threshold < 1 and self.n_iters_no_change >= self.stop_threshold

        if reached_stop_threshold() or reached_ensemble_size():
            self.base.coef_, self.base.intercept_ = self.pocket_best
            return True
        else:
            return False

    # TRAINING
    @validate_call
    def train(
        self,
        ensemble_size: Annotated[int, Field(gt=1)] = 50,
        epoch_size: Annotated[float, Field(gt=0, le=1)] = 1.0,
        data_opts: Literal["partial", "cycle", "window", "whole"] = "whole",
        stop_threshold: int = 50,
        log: Literal["all", "mid", "min", "none"] = "mid",
        write: bool = True,
        outfile: str = "training_runs.csv",
    ) -> None:
        """
        Trains the perceptron model.

        ### Parameters
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

        stop_threshold (int):
            The number of epochs to wait before stopping training if no improvement is seen.
            < 1: No early stopping. Will train the full ensemble_size

        log (Literal["all", "min", "none"]):
            The level of logging during training.
            "all": Log all information.
            "min": Log minimal information.
            "none": Do not log any information.

        write (bool):
            Whether to write the results to a file.

        outfile (str):
            The file to save the results to.

        ### Returns
            None
        """

        with self._training_context(locals()):  # set training options and start timer
            # infinite data according to data_opts
            data = self._data_generator()

            # loop over ensemble of perceptrons trained for one epoch each
            for weak_learner in self._train_ensemble_parallel(data):  # swap the training function for parallel or sequential
                self._add_weights(weak_learner)  # add weights of new weak learner
                self._test_and_record()  # get acc of updated ensemble

                if self.log == "all":
                    print(self.dev_accs[-1])

                if self._pocket_stop():
                    break

    def _test_and_record(self) -> float:
        self.dev_accs.append(self.dev_accuracy)
        self.test_accs.append(self.accuracy)

    def _train_ensemble_parallel(self, data) -> Generator[skPerc, None, None]:
        with ThreadPoolExecutor() as executor:
            while True:
                futures = []
                for _ in range(5):  # submit 5 at a time to accomidate early stopping
                    futures.append(executor.submit(self._train_weak_learner, *next(data)))

                for future in as_completed(futures):
                    yield future.result()

    def _train_ensemble_sequential(self, data) -> Generator[skPerc, None, None]:
        while True:
            yield self._train_weak_learner(*next(data))

    def _train_weak_learner(self, X, y) -> skPerc:
        return skPerc().partial_fit(X, y, classes=self.base.classes_)

    def _add_weights(self, new: skPerc) -> None:
        self.base.coef_ = self.base.coef_ + new.coef_
        self.base.intercept_ = self.base.intercept_ + new.intercept_

    def _write_results(self, outfile) -> None:
        with open(outfile, "a") as f:
            data = {
                "dataset": self.dataset_name,
                "accuracy": self.accuracy,
                "ensemble_size": self.ensemble_size,
                "epoch_size": self.epoch_size,
                "data_option": self.data_opts,
                "train_time": self.train_time,
                "dev_max_at": self.dev_max_at,
                "test_max_at": self.test_max_at,
                "num_iters": self.n_iters,
                "test_accs": self.test_accs,
                "dev_accs": self.dev_accs,
            }
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if os.path.getsize(outfile) == 0:  # write header if file is empty
                writer.writeheader()
            writer.writerow(data)

    def print_info(self, include: list[str] = None) -> None:
        data = {
            "DATASET": [
                ("Dataset", self.dataset_name),
                ("Train size", self.train_size),
                ("Test size", self.test_size),
                ("Dev size", self.dev_size),
                ("Classes", self.n_classes),
                ("Features", self.n_features),
            ],
            "RUN INFO": [
                ("Ensemble", self.ensemble_size),
                ("Stopped at", self.n_iters),
                ("Epoch size", self.epoch_size),
                ("Data opts", self.data_opts),
                ("Train time", self.train_time_string),
                ("Max at", self.dev_max_at),
                ("Accuracy", self.accuracy_string),
            ],
        }
        # loop over data and print
        for section, info in data.items():
            if not include or section.upper() in include:
                print(f"\n{section:-^20}")
                for k, v in info:
                    print(f"{k+':':<12}{v}")

    def plot(self):
        plt.plot(self.dev_accs)
        plt.plot(self.test_accs)
        plt.title(f"{self.dataset_name}")
        plt.xlabel("Ensemble Size")
        plt.ylabel("Accuracy")
        plt.legend(["Dev", "Test"])
        plt.ylim(0.5, 1)

        # info box
        props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
        textstr = f"Test Acc: {self.accuracy}\nMax at: {self.dev_max_at}"
        plt.text(0.75, 1.15, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment="top", bbox=props)

        plt.show()


if __name__ == "__main__":
    ...
