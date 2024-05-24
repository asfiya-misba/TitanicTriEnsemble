# Asfiya Misba - 1002028239
# Summer 2023
# Assignment 3
import numpy as np


class DecisionTree:
    def __init__(self, criterion, max_depth, min_sample_split, min_samples_leaf, sample_feature=False):
        if criterion == 'misclassification':
            self.criterion = self.misclassification
        elif criterion == 'gini':
            self.criterion = self.gini_gain
        elif criterion == 'entropy':
            self.criterion = self.information_gain
        else:
            raise Exception('Criterion should be misclassification or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    # To train the model
    def fit(self, X, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)
        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self.build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    # To calculate misclassification
    @staticmethod
    def misclassification(X, y, index, sample_weights=None):
        misclassification_rate = 0
        total_labels = y.shape[0]
        labels = {}
        for i in range(total_labels):
            if y[i] not in labels.keys():
                labels[y[i]] = 0
            labels[y[i]] += sample_weights[i]
        majority_label = max(labels, key=labels.get)
        misclassification_rate = 1.0 - labels[majority_label] / np.sum(sample_weights)
        return misclassification_rate

    # To calculate entropy
    @staticmethod
    def entropy(y, sample_weights=None):
        total_labels = y.shape[0]
        labels = {}
        entropy_value = 0.0
        for i in range(total_labels):
            if y[i] not in labels.keys():
                labels[y[i]] = 0
            labels[y[i]] += sample_weights[i]
        for key in labels:
            prob = float(labels[key]) / float(np.sum(sample_weights))
            entropy_value -= prob * np.log2(prob)
        return entropy_value

    # To calculate information gain
    def information_gain(self, X, y, index, sample_weights=None):
        new_value = 0.0
        gain_value = 0
        old_value = self.entropy(y, sample_weights)
        unique_val = np.unique(X[:, index])
        for value in unique_val:
            sub_X, sub_y, sub_sample_weights = self.make_subsets(X, y, index, value, sample_weights)
            prob = np.sum(sub_sample_weights) / float(np.sum(sample_weights))
            new_value += prob * self.entropy(sub_y, sub_sample_weights)
        gain_value = old_value - new_value
        return gain_value

    @staticmethod
    def gini_impurity_value(y, sample_weights=None):
        gini = 1
        total_labels = y.shape[0]
        label = {}
        for i in range(total_labels):
            if y[i] not in label.keys():
                label[y[i]] = 0
            label[y[i]] += sample_weights[i]
        for key in label:
            probability = float(label[key]) / float(np.sum(sample_weights))
            gini -= probability ** 2
        return gini

    def gini_gain(self, X, y, index, sample_weights=None):
        new_value = 0.0
        old_value = self.gini_impurity_value(y, sample_weights)
        gini_value = 0
        unique_val = np.unique(X[:, index])
        for value in unique_val:
            sub_X, sub_y, sub_sample_weights = self.make_subsets(X, y, index, value, sample_weights)
            prob = np.sum(sub_sample_weights) / float(np.sum(sample_weights))
            new_value += prob * self.gini_impurity_value(sub_y, sub_sample_weights)
        gini_value = old_value - new_value
        return gini_value

    # To make subsets of the tree
    def make_subsets(self, X, y, index, value, sample_weights=None):
        features = X[:, index]
        X = X[:, [i for i in range(X.shape[1]) if i != index]]
        match = []
        for i in range(len(features)):
            if features[i] == value:
                match.append(i)
        sub_X = X[match, :]
        sub_y = y[match]
        sub_sample_weights = sample_weights[match]
        return sub_X, sub_y, sub_sample_weights

    # To make predictions
    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("Fit method not called")

        def traverse_and_classify(tree, x):
            feature_name = list(tree.keys())[0]
            sub_tree = tree[feature_name]
            key = x.loc[feature_name]
            if key not in sub_tree:
                key = np.random.choice(list(sub_tree.keys()))
            valueOfKey = sub_tree[key]
            if isinstance(valueOfKey, dict):
                label = traverse_and_classify(valueOfKey, x)
            else:
                label = valueOfKey
            return label

        # If only one sample is passed, classify it directly
        if len(X.shape) == 1:
            return traverse_and_classify(self._tree, X)
        else:
            # If multiple samples are passed, classify each sample and store the results in a list
            results = []
            for i in range(X.shape[0]):
                results.append(traverse_and_classify(self._tree, X.iloc[i, :]))
            return np.array(results)

    # To choose the best feature
    def best_feature(self, X, y, sample_weights=None):
        n_features = X.shape[1]
        if self.sample_feature:
            max_features = max(1, min(n_features, int(np.round(np.sqrt(n_features)))))
            new_features = np.random.choice(n_features, max_features, replace=False)
            new_X = X[:, new_features]
        else:
            new_X = X
        n_new_features = new_X.shape[1]
        best_gain = 0.0
        feature_index = 0
        for i in range(n_new_features):
            gain_cost = self.criterion(new_X, y, i, sample_weights)
            if gain_cost > best_gain:
                best_gain = gain_cost
                feature_index = i
        return feature_index

    # To find the majority
    @staticmethod
    def majority_vote(y, sample_weights=None):
        majority = y[0]
        label_weights = {}
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        for i in range(y.shape[0]):
            if y[i] not in label_weights.keys():
                label_weights[y[i]] = sample_weights[i]
            else:
                label_weights[y[i]] += sample_weights[i]
        majority = max(label_weights, key=label_weights.get)
        return majority

    # To build the decision tree
    def build_tree(self, X, y, feature_names, depth, sample_weights=None):
        mytree = dict()
        if len(feature_names) == 0 or len(np.unique(y)) == 1 or depth >= self.max_depth or len(
                X) <= self.min_samples_leaf:
            return self.majority_vote(y, sample_weights)
        if len(X) >= self.min_sample_split:
            best_feature_idx = self.best_feature(X, y, sample_weights)
            best_feature_name = feature_names[best_feature_idx]
            feature_names = feature_names[:]
            feature_names.remove(best_feature_name)
            mytree = {best_feature_name: {}}
            unique_vals = np.unique(X[:, best_feature_idx])
            for value in unique_vals:
                sub_X, sub_y, sub_sample_weights = self.make_subsets(X, y, best_feature_idx, value, sample_weights)
                if len(sub_X) >= self.min_sample_split:
                    mytree[best_feature_name][value] = self.build_tree(sub_X, sub_y, feature_names, depth + 1,
                                                                       sub_sample_weights)
                else:
                    mytree[best_feature_name][value] = self.majority_vote(sub_y, sub_sample_weights)
        else:
            # Making it leaf node if samples are less than min_sample_split
            return self.majority_vote(y, sample_weights)
        return mytree

    # To print the decision tree
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self._tree
        if isinstance(node, dict):
            for feature_name, sub_tree in node.items():
                print("  " * depth + f"{feature_name}:")
                for value, sub_node in sub_tree.items():
                    print("  " * (depth + 1) + f"if {feature_name} = {value}:")
                    self.print_tree(sub_node, depth + 2)
        else:
            print("  " * (depth + 1) + f"-> Class: {node}")
