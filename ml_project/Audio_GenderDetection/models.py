"""
This module implements various algorithms for gender classification task. The API must be similar for all models
"""
import joblib
from abc import ABC, abstractmethod

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

import os


class GenderDetectionModel(ABC):
    @abstractmethod
    def __init__(self):
        self.model = None
        self.name = None

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, metric: str = "accuracy"):
        y_pred = self.predict(x)
        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        elif metric == "f1":
            return f1_score(y, y_pred)
        elif metric == "precision":
            return precision_score(y, y_pred)
        elif metric == "recall":
            return recall_score(y, y_pred)
        else:
            raise ValueError("Metric must be either 'accuracy' or 'f1' or 'precision' or 'recall'")

    def cross_validate(self, x, y, folds=5):
        scores = cross_val_score(self.model, x, y, cv=folds)
        return scores

    def save_model(self, directory: str = "models"):
        joblib.dump(self.model, os.path.join(directory, f"{self.name}_model.joblib"))
        print(f"Model saved as {os.path.join(directory, f'{self.name}_model.joblib')}")


class SupportVectorMachine(GenderDetectionModel):
    def __init__(self, probability: bool = True):
        self.probability = probability
        self.model = SVC(probability=probability)
        self.name = "support_vector_machine"


class KNearestNeighbors(GenderDetectionModel):
    def __init__(self):
        self.model = KNeighborsClassifier()
        self.name = "knn"


class AdaBoost(GenderDetectionModel):
    def __init__(self):
        self.model = AdaBoostClassifier()
        self.name = "adaboost"


class MultiLayerPerceptron(GenderDetectionModel):
    def __init__(self, max_iter: int = 1000):
        self.model = MLPClassifier(max_iter=max_iter)
        self.name = "multilayer_perceptron"


class LogisticRegression(GenderDetectionModel):
    def __init__(self, max_iter: int = 1000):
        self.model = LogReg(max_iter=max_iter)
        self.name = "logistic_regression"
