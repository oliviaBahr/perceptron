from sklearn.linear_model import Perceptron
import random

class AddPerceptron:
    
    def __init__(self, **kwargs):
        """Olivia's additive Perceptron implemented through sklearn."""
        self.max_iter = kwargs.get('max_iter', 100) # Store passed max_iter since we rig the actual one to 1
        random.seed(a = kwargs.get('random_state', 42))
        kwargs['max_iter'] = 1
        self.clf = Perceptron(**kwargs)

    def fit(self, X, y):
        self.clf.fit(X, y)
        for i in range(self.max_iter - 1):
            clfp = Perceptron(max_iter = 1, random_state = random.randint(0,100000000))
            clfp.fit(X, y)
            self.clf.coef_ += clfp.coef_
            self.clf.intercept_ += clfp.intercept_

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, *args, **kwargs):
        return self.clf.score(*args, **kwargs)
