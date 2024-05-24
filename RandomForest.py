# Asfiya Misba - 1002028239
# Summer 2023
# Assignment 3
import copy
import numpy as np
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool


class RandomForest:
    def __init__(self, base_learner, n_estimator, min_features, seed=0):
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self.min_features = min_features
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]

    # Sampling with replacement
    def sampling_with_replacement(self, X, y, i):
        L = y.shape[0]
        index = np.random.choice(L, L, replace=True)
        X_sample, y_sample = X.iloc[index, :], y.iloc[index]
        self._estimators[i].fit(X_sample, y_sample)

    # Randomly selecting features
    def selecting_features(self, num_features):
        num_selected_features = np.random.randint(self.min_features, num_features + 1)
        return np.random.choice(num_features, num_selected_features, replace=False)

    # To train the model
    def fit(self, X, y):
        self.labels = list(set(y))
        num_features = X.shape[1]
        if self.min_features > num_features:
            raise ValueError('min_features cannot be greater than the total number of features')
        sampling = partial(self.sampling_with_replacement, X, y)
        pool = ThreadPool(self.n_estimator)
        sampled_sets = pool.map(sampling, list(range(self.n_estimator)))
        pool.close()
        pool.join()
        for i, data in enumerate(sampled_sets):
            if data is not None:  # ignore the iteration when the data is none
                X_sample, y_sample = data
                self._estimators[i].fit(X_sample, y_sample)
        return self

    # To make predictions
    def predict(self, X):
        N = X.shape[0]
        y_pred = np.zeros(N)
        predictions = np.array([estimator.predict(X) for estimator in self._estimators])
        pred_probability = np.array([(predictions == label).mean(axis=0) for label in self.labels]).T
        y_pred = np.array(self.labels)[np.argmax(pred_probability, axis=1)]
        return y_pred
