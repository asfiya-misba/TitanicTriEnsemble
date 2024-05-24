# Asfiya Misba - 1002028239
# Summer 2023
# Assignment 3
import copy
import numpy as np


class Adaboost:
    def __init__(self, weak_learner, num_learners, learning_rate=1.0, seed=0):
        np.random.seed(seed)
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.learners = [copy.deepcopy(self.weak_learner) for _ in range(self.num_learners)]
        self.learner_weights = [1 for _ in range(self.num_learners)]

    # To train the model
    def fit(self, X, y):
        L = y.shape[0]
        w = np.ones(L) / L
        for i in range(self.num_learners):
            self.learners[i].fit(X, y, sample_weights=w)
            pred = self.learners[i].predict(X)
            err = np.sum((pred != y) * w)
            if err == 0:
                self.num_learners = i + 1  # Stop early if a perfect fit is achieved
                break
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)
            self.learner_weights[i] = alpha
            w *= np.exp(-alpha * np.array(y) * pred) + 1e-4
            w /= 2 * np.sqrt(err * (1 - err))
        return self

    # To make predictions
    def predict(self, X):
        N = X.shape[0]
        y_pred = np.zeros(N)
        for i in range(self.num_learners):
            y_pred += self.learner_weights[i] * self.learners[i].predict(X)
        y_pred = np.sign(y_pred)
        return y_pred

