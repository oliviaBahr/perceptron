from functools import lru_cache
from os import listdir
from os.path import abspath, join

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, spmatrix
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle as sklearn_shuffle


@lru_cache
def load_dir(
    path: str, multilabel: bool = False
) -> tuple[tuple[spmatrix | ndarray, ndarray], tuple[spmatrix | ndarray, ndarray]]:
    paths = [abspath(join(path, p)) for p in listdir(path)]
    if "imdb" in path.lower():
        return _load_imdb_binary(paths)  # type: ignore

    match len(paths):
        case 1:
            (X, y), (tX, ty) = _load_split(paths[0], multilabel=multilabel)
        case 2:
            if paths[0].endswith(".t"):
                paths = paths[::-1]
            X, y, *_ = load_svmlight_file(paths[0], multilabel=multilabel)
            tX, ty, *_ = load_svmlight_file(paths[1], multilabel=multilabel)
        case _:
            raise ValueError(f"Expected 1 or 2 files in {path}, got {len(paths)}")
    return (X, y), (tX, ty)


@lru_cache
def load_file(path: str, multilabel: bool = False) -> tuple[spmatrix | ndarray, ndarray]:
    if "imdb" in path.lower():
        raise ValueError("Use load_dir() for the imdb dataset")
    else:
        X, y, *_ = load_svmlight_file(path, multilabel=multilabel)
        return X, y


def dev_split(
    X: spmatrix | ndarray, y: ndarray, dev_size=0.1
) -> tuple[tuple[spmatrix | ndarray, ndarray], tuple[spmatrix | ndarray, ndarray]]:
    X, dX, y, dy = train_test_split(X, y, test_size=dev_size, random_state=0)
    return (X, y), (dX, dy)


def shuffle(X: spmatrix | ndarray, y: ndarray) -> tuple[spmatrix | ndarray, ndarray]:
    return sklearn_shuffle(X, y)  # type: ignore


def resample_if(X: spmatrix | ndarray, y: ndarray, epoch_size: float) -> tuple[spmatrix | ndarray, ndarray]:
    """Resample part of data if epoch_size < 1.0."""
    if epoch_size < 1.0:
        return tuple(resample(X, y, replace=False, n_samples=int(X.shape[0] * epoch_size)))  # type: ignore
    return X, y


def _load_split(
    path: str, test_size=None, multilabel: bool = False
) -> tuple[tuple[ndarray | spmatrix, ndarray], tuple[ndarray | spmatrix, ndarray]]:
    data = load_svmlight_file(path, multilabel=multilabel)
    X, tX, y, ty = train_test_split(data[0], data[1], test_size=test_size, random_state=0)
    return (X, y), (tX, ty)


def _load_imdb_binary(paths: list[str]) -> tuple[tuple[spmatrix, ndarray], tuple[spmatrix, ndarray]]:
    if len(paths) != 2:
        raise ValueError(f"Expected 2 files, got {len(paths)}: {paths}")

    trainpath, testpath = paths[0], paths[1]
    res_X = []  # type: ignore
    res_tX = []  # type: ignore
    res_y: list[int] = []
    res_ty: list[int] = []

    featfunc = lambda x: 1 if x > 0 else 0  # make features binary
    classfunc = lambda x: 1 if x > 4 else 0  # make classes binary
    popClass = lambda x: int(x.pop(0))
    splitItem = lambda x: x.split(":")

    with open(trainpath) as trainfile, open(testpath) as testfile:

        for line in trainfile:
            data = line.strip().split()
            res_y.append(classfunc(popClass(data)))
            res_X.append({i: featfunc(float(v)) for i, v in map(splitItem, data)})

        for line in testfile:
            data = line.strip().split()
            res_ty.append(classfunc(popClass(data)))
            res_tX.append({i: featfunc(float(v)) for i, v in map(splitItem, data)})

    vectorizer = DictVectorizer()
    Xy = csr_matrix(vectorizer.fit_transform(res_X)), np.array(res_y)
    tXy = csr_matrix(vectorizer.transform(res_tX)), np.array(res_ty)
    return Xy, tXy
