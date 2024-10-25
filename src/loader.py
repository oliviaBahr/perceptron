from functools import lru_cache
from numpy import ndarray
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle

class Loader:
    @staticmethod
    def get_name(pathname) -> str:
        return pathname.split("/")[-2].lower()

    @staticmethod
    @lru_cache
    def load(trainpath, testpath=None, test_size=None) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        if "imdb" in trainpath.lower():
            return Loader._load_imdb_binary(trainpath, testpath)
        elif not testpath:
            return Loader._load_split(trainpath, test_size)
        else:
            return load_svmlight_file(trainpath), load_svmlight_file(testpath)

    @staticmethod
    @lru_cache
    def dev_split(data, dev_size=.1) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        X, dX, y, dy = train_test_split(data[0], data[1], test_size=dev_size, random_state=0)
        return (X, y), (dX, dy)
    
    @staticmethod
    @lru_cache
    def shuffle(X, y) -> tuple[ndarray, ndarray]:
        return sklearn_shuffle(X, y)

    @staticmethod
    @lru_cache
    def conditional_shuffle(X, y, condition) -> tuple[ndarray, ndarray]:
        return sklearn_shuffle(X, y) if condition else (X, y)

    @staticmethod
    @lru_cache
    def _load_split(trainpath, test_size=None) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        data = load_svmlight_file(trainpath)
        X, tX, y, ty = train_test_split(data[0], data[1], test_size=test_size, random_state=0)
        return (X, y), (tX, ty)

    @staticmethod
    @lru_cache
    def _load_imdb_binary(trainpath, testpath) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        res_X, res_y, res_tX, res_ty = [], [], [], []

        featfunc = lambda x: 1 if x > 0 else 0 # make features binary
        classfunc = lambda x: 1 if x > 4 else 0 # make classes binary
        popClass = lambda x: int(x.pop(0))
        splitItem = lambda x: x.split(":")

        for path, X, y in [(trainpath, res_X, res_y), (testpath, res_tX, res_ty)]:
            for line in open(path):
                line = line.strip().split()
                y.append(classfunc(popClass(line))) 
                X.append({int(i): featfunc(float(v)) for i, v in map(splitItem, line)})

        vectorizer = DictVectorizer()
        res_X = vectorizer.fit_transform(res_X)
        res_tX = vectorizer.transform(res_tX)
        return (res_X, res_y), (res_tX, res_ty)
