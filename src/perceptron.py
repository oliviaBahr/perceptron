import warnings, csv, os
from copy import deepcopy
from contextlib import contextmanager
from typing import Literal, Iterator, Generator, Annotated
from pydantic import validate_call, Field
from scipy.sparse import spmatrix, csr_matrix, vstack, hstack
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import lru_cache
from itertools import cycle
from numpy import ndarray, unique
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
    def load(path1, path2) -> tuple[spmatrix, ndarray]:
        return load_svmlight_file(path1), load_svmlight_file(path2)

    @staticmethod
    @lru_cache
    def load_split(path1, test_size=None) -> tuple[spmatrix, ndarray]:
        data = load_svmlight_file(path1)
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
        # base perceptron to store weights
        self.base = None

        # data
        self.dataset_name = dataset_name if dataset_name else Loader.get_name(trainpath)

        if "imdb" in self.dataset_name.lower():
            traindata, testdata = Loader.load_imdb_binary(trainpath, testpath)
        elif testpath:
            traindata, testdata = Loader.load(trainpath, testpath)
        else:
            traindata, testdata = Loader.load_split(trainpath, test_size)

        self.X, self.devX, self.y, self.devy = train_test_split(*traindata, test_size=0.1)
        self.tX, self.ty = testdata

        # dataset info
        self.train_size = self.X.shape[0]
        self.test_size = self.tX.shape[0]
        self.dev_size = self.devX.shape[0]
        self.n_classes = len(unique(self.y))
        self.n_features = self.X.shape[1]
        self.n_weights = 1 if self.n_classes == 2 else self.n_classes

        # training run info
        self.accuracy = None
        self.test_accs = None
        self.dev_accs = None
        self.epoch_size = None
        self.ensemble_size = None
        self.data_opts = None
        self.train_time = None

        # logging and writing options
        self.outfile = None
        self.write = None
        self.log = None

    def test(self) -> float:
        """Tests the model and returns the accuracy."""
        acc = self.base.score(self.tX, self.ty)
        return float(f"{acc:.5f}")

    # wraps the training
    @contextmanager
    def _training_context(self, train_opts: dict):
        try:
            # reset base perceptron and accuracies
            self.base = skPerc(max_iter=1).fit(self.X, self.y)  # fit to get the right shapes for coef_ and intercept_
            self.base.coef_.fill(1)
            self.base.intercept_.fill(0)
            self.accuracy = 0
            self.test_accs = []
            self.dev_accs = []

            # set opts as object attributes ("data_opts", "ensemble_size", etc.)
            del train_opts["self"]
            for key, value in train_opts.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # log
            if self.log != "none":
                print(f"Training {self.ensemble_size} learners on {self.dataset_name} ({self.data_opts}, {self.epoch_size})...")

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

            # set final info
            self.accuracy = self.test()

            # write to csv
            if self.write:
                self._write_results(self.outfile)

            # log info
            if self.log != "none":
                print(f"Training time: {self.train_time_str()}")
                if self.log == "min":
                    if len(self.dev_accs) < self.ensemble_size:
                        print(f"Stopped early at ensemble size {len(self.dev_accs)}")
                    print(f"Accuracy: {self.accuracy}\n")
                else:
                    self.print_info()
        finally:
            spinner.stop()

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
            # early stopping trackers
            best = [self.base.coef_, self.base.intercept_]
            n_iter_no_change = 0

            # infinite data according to data_opts
            data = self._data_generator()

            # loop over ensemble of perceptrons trained for one epoch each
            for weak_learner in self._train_ensemble_parallel(data):  # swap the training function for parallel or sequential
                self._add_weights(weak_learner)  # add weights of new weak learner
                self._test_and_record()  # get acc of updated ensemble
                if self.log == "all":
                    print(self.dev_accs[-1])

                # pocket early stopping
                if stop_threshold > 0 and len(self.dev_accs) > 1:
                    if self.dev_accs[-1] > max(self.dev_accs[:-1]):
                        n_iter_no_change = 0
                        best = [self.base.coef_, self.base.intercept_]
                    elif n_iter_no_change < stop_threshold:
                        n_iter_no_change += 1
                    else:
                        self.base.coef_, self.base.intercept_ = best
                        break

    def _test_and_record(self) -> float:
        dev = self.base.score(self.devX, self.devy)
        test = self.base.score(self.tX, self.ty)
        self.dev_accs.append(dev)
        self.test_accs.append(test)

    def _train_ensemble_parallel(self, data) -> Generator[skPerc, None, None]:
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(self.ensemble_size):
                futures.append(executor.submit(self._train_weak_learner, *next(data)))

            for future in as_completed(futures):
                yield future.result()

    def _train_ensemble_sequential(self, data) -> Generator[skPerc, None, None]:
        for _ in range(self.ensemble_size):
            yield self._train_weak_learner(*next(data))

    def _train_weak_learner(self, X, y) -> skPerc:
        return skPerc().partial_fit(X, y, classes=self.base.classes_)

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
                        ywin = hstack([y[i:], y[: end - self.train_size]]).T.toarray()
                        windows.append((Xwin, ywin))

                return cycle(windows)

            case _:
                raise ValueError("data_opts must be 'partial', 'cycle', 'window', or 'whole'")

    def _write_results(self, outfile) -> None:
        with open(outfile, "a") as f:
            data = {
                "dataset": self.dataset_name,
                "accuracy": float(f"{self.accuracy:.5f}"),
                "ensemble_size": self.ensemble_size,
                "epoch_size": self.epoch_size,
                "data_option": self.data_opts,
                "train_time": self.train_time,
                "dev_max_at": self.dev_accs.index(max(self.dev_accs)) + 1,
                "test_max_at": self.test_accs.index(max(self.test_accs)) + 1,
                "num_iters": len(self.dev_accs),
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
                ("Stopped at", len(self.dev_accs)),
                ("Epoch size", self.epoch_size),
                ("Data opts", self.data_opts),
                ("Train time", self.train_time_str()),
                ("Max at", self.dev_accs.index(max(self.dev_accs)) + 1),
                ("Accuracy", self.accuracy),
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
        plt.plot(self.dev_accs)
        plt.plot(self.test_accs)
        plt.title(f"{self.dataset_name}")
        plt.xlabel("Ensemble Size")
        plt.ylabel("Accuracy")
        plt.legend(["Dev", "Test"])
        plt.ylim(0.5, 1)

        # info box
        props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
        textstr = f"Test Acc: {self.accuracy}\nMax at: {self.dev_accs.index(max(self.dev_accs)) + 1}"
        plt.text(0.75, 1.15, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment="top", bbox=props)

        plt.show()


if __name__ == "__main__":
    ...
