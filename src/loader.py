from scipy.sparse import spmatrix
from functools import lru_cache
from numpy import ndarray
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


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
        return (X, y), (tX, ty)
