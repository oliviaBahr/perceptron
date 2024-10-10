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

        fig = variance_plots(run, classifiers, data_name)
        run["variance_plot"].upload(fig)
        run.stop()


def variance_plots(run, classifiers, data_name) -> plt.Figure:
    run.wait()
    clf_names = [name for name, _ in classifiers]

    scores = [run[f"{clf_name}/acc"].fetch_values().value.tolist() for clf_name in clf_names]
    scores = np.array(scores).T

    iter_counts = [run[f"{clf_name}/iters"].fetch_values().value.tolist() for clf_name in clf_names]
    iter_counts = np.array(iter_counts).T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(data_name)

    ax1.violinplot(scores, showmeans=True)
    ax1.set_xticks(np.arange(1, len(clf_names) + 1))
    ax1.set_xticklabels(clf_names)
    ax1.set_ylabel("Accuracy")

    ax2.set_xticks(np.arange(1, len(clf_names) + 1))
    ax2.set_xticklabels(clf_names)
    ax2.set_ylabel("Iterations")
    parts = ax2.violinplot(iter_counts, showmeans=True)
    _ = [pc.set_facecolor("red") for pc in parts["bodies"]]
    _ = [parts[bar].set_color("red") for bar in ["cmaxes", "cmins", "cbars", "cmeans"]]

    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    CLASSIFIERS = [
        ("perc", Perceptron),
        ("add_perc", addperceptron.AddPerceptron),
        ("sgd", SGDClassifier),
        ("add_sgd", addsgdclassifier.AddSGDClassifier),
        # ("mlp", MLPClassifier),
        # ("add_mlp", addmlpclassifier.AddMLPClassifier),
    ]

    DATA_PATHS = [
        ("imdb", "./data/imdb/train_labeledBow.feat", "./data/imdb/test_labeledBow.feat"),
        # ("cod-rna", "../data/cod-rna/train.csv", "../data/cod-rna/test.csv"),
    ]

    N_RUNS = 10
    start(N_RUNS, CLASSIFIERS, DATA_PATHS)
