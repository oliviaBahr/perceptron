from functools import lru_cache

import numpy as np
from numpy import ndarray
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle as sklearn_shuffle


class Loader:
    @staticmethod
    def get_name(pathname: str) -> str:
        return pathname.split("/")[-2].lower()

    @staticmethod
    @lru_cache
    def load(trainpath, testpath=None, test_size=None) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        if "imdb" in trainpath.lower():
            if not testpath:
                raise ValueError("testpath is required for imdb dataset")
            return Loader._load_imdb_binary(trainpath, testpath)
        elif not testpath:
            return Loader._load_split(trainpath, test_size)
        else:
            return load_svmlight_file(trainpath), load_svmlight_file(testpath)

    @staticmethod
    def dev_split(X, y, dev_size=0.1) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        X, dX, y, dy = train_test_split(X, y, test_size=dev_size, random_state=0)
        return (X, y), (dX, dy)

    @staticmethod
    def shuffle(X, y) -> tuple[ndarray, ndarray]:
        return sklearn_shuffle(X, y)

    @staticmethod
    def resample_if(X, y, epoch_size) -> tuple[ndarray, ndarray]:
        """Resample part of data if epoch_size < 1.0."""
        if epoch_size < 1.0:
            return resample(X, y, replace=False, n_samples=int(X.shape[0] * epoch_size))
        return X, y

    @staticmethod
    @lru_cache
    def _load_split(trainpath, test_size=None) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        data = load_svmlight_file(trainpath)
        X, tX, y, ty = train_test_split(data[0], data[1], test_size=test_size, random_state=0)
        return (X, y), (tX, ty)

    @staticmethod
    @lru_cache
    def _load_imdb_binary(trainpath: str, testpath: str) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        res_X: list[dict[int, int]] = []
        res_tX: list[dict[int, int]] = []
        res_y: list[int] = []
        res_ty: list[int] = []

        featfunc = lambda x: 1 if x > 0 else 0  # make features binary
        classfunc = lambda x: 1 if x > 4 else 0  # make classes binary
        popClass = lambda x: int(x.pop(0))
        splitItem = lambda x: x.split(":")

        for path, X, y in [(trainpath, res_X, res_y), (testpath, res_tX, res_ty)]:
            for line in open(path):
                data = line.strip().split()
                y.append(classfunc(popClass(data)))
                X.append({int(i): featfunc(float(v)) for i, v in map(splitItem, data)})

        vectorizer = DictVectorizer()
        Xy = vectorizer.fit_transform(res_X), np.array(res_y)
        tXy = vectorizer.transform(res_tX), np.array(res_ty)
        return Xy, tXy
