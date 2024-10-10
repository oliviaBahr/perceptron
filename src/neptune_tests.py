import neptune
from sklearn.utils import shuffle
import addperceptron, addmlpclassifier, addsgdclassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from perceptron import Loader
from tqdm import trange


def start(run, data_paths: tuple[str, str], classifiers):
    print("Starting Neptune run")
    for data_name, train_path, test_path in data_paths:
        (X, y), (tX, ty) = Loader.load(train_path, test_path)
        clf_name = ""
        for clf_name, Clf in classifiers:
            for _ in trange(50, desc=clf_name):
                X, y = shuffle(X, y)
                clf = Clf()
                clf.fit(X, y)
                run[f"{data_name}/{clf_name}/acc"].append(clf.score(tX, ty))
                run[f"{data_name}/{clf_name}/iters"].append(clf.n_iter_)


if __name__ == "__main__":
    RUN = neptune.init_run(project="adding-perceptron/variance-experiments")

    CLASSIFIERS = [
        ("perceptron", Perceptron),
        ("mlp", MLPClassifier),
        ("sgd", SGDClassifier),
        ("add_perceptron", addperceptron.AddPerceptron),
        ("add_mlp", addmlpclassifier.AddMLPClassifier),
        ("add_sgd", addsgdclassifier.AddSGDClassifier),
    ]

    DATA_PATHS = [
        ("imdb", "./data/imdb/train_labeledBow.feat", "./data/imdb/test_labeledBow.feat"),
        # ("cod-rna", "../data/cod-rna/train.csv", "../data/cod-rna/test.csv"),
    ]

    start(RUN, DATA_PATHS, CLASSIFIERS)
    RUN.stop()
