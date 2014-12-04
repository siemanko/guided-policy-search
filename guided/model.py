import numpy as np

class Model(object):
    def fit(X, Y, weights):
        return self._fit(X, Y, weights)

    def predict(X):
        return self._predict(X)

    def controller(self):
        def c(x,t):
            if any(np.isnan(x)) or not all(np.abs(x) < 1e100):
                return 0.0
            return self.predict(np.array([x]))[0]
        return c

class FeatureModel(Model):
    def __init__(self):
        self.features = None

    def fit(self, X, Y, weights):
        if self.features:
            X = self.features(X)
        return self._fit(X, Y, weights)

    def predict(self, X):
        if self.features:
            X = self.features(X)
        return self._predict(X)


class SklearnModel(FeatureModel):
    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model

    def _fit(self, X, Y, weights):
        return self.sklearn_model.fit(X, Y, weights)

    def _predict(self, x):
        return self.sklearn_model.predict(x)
