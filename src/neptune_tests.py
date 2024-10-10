import numpy as np
import matplotlib.pyplot as plt
import neptune
from sklearn.utils import shuffle
import addperceptron, addmlpclassifier, addsgdclassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from perceptron import Loader
from tqdm import trange


def start(n_runs, classifiers, data_paths):
    print("Starting Neptune run")
    for data_name, train_path, test_path in data_paths:
        run = neptune.init_run(project="adding-perceptron/variance-experiments")
        run["sys/tags"].add([data_name])
        (X, y), (tX, ty) = Loader.load(train_path, test_path)

        for clf_name, Clf in classifiers:
            for _ in trange(n_runs, desc=clf_name):
                X, y = shuffle(X, y)
                clf = Clf()
                clf.fit(X, y)
                run[f"{clf_name}/acc"].append(accuracy_score(ty, clf.predict(tX)))
                run[f"{clf_name}/iters"].append(clf.n_iter_)

        run.wait()
        scores = [run[f"{clf_name}/acc"].fetch_values().value.tolist() for clf_name, _ in classifiers]

        fig = plot(classifiers, data_name, scores)
        run["variance_plot"].upload(fig)
        run.stop()


def plot(classifiers, data_name, scores) -> plt.Figure:
    clf_names = [name for name, _ in classifiers]
    scores = np.array(scores).T

    fig, ax = plt.subplots()
    ax.violinplot(scores, showmeans=True)
    ax.set_xticks(np.arange(1, len(clf_names) + 1))
    ax.set_xticklabels(clf_names)
    ax.set_ylabel("Accuracy")
    ax.set_title(data_name)
    return fig


if __name__ == "__main__":
    CLASSIFIERS = [
        ("perc", Perceptron),
        ("add_perc", addperceptron.AddPerceptron),
        ("sgd", SGDClassifier),
        ("add_sgd", addsgdclassifier.AddSGDClassifier),
        ("mlp", MLPClassifier),
        ("add_mlp", addmlpclassifier.AddMLPClassifier),
    ]

    DATA_PATHS = [
        ("imdb", "./data/imdb/train_labeledBow.feat", "./data/imdb/test_labeledBow.feat"),
        # ("cod-rna", "../data/cod-rna/train.csv", "../data/cod-rna/test.csv"),
    ]

    N_RUNS = 10
    start(N_RUNS, CLASSIFIERS, DATA_PATHS)
