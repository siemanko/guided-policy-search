import numpy as np
from numpy import pi

class Model(object):
    def fit(self,X, Y, weights):
        return self._fit(X, Y, weights)

    def predict(self,X):
        X[:, 0] %= 2*pi
        X[:, 2] %= 2*pi
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
        X[:, 0] %= 2*pi
        X[:, 2] %= 2*pi
        return self._predict(X)


class SklearnModel(FeatureModel):
    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model

    def _fit(self, X, Y, weights):
        return self.sklearn_model.fit(X, Y)

    def _predict(self, x):
        return self.sklearn_model.predict(x)

class GmmModel(FeatureModel):
    def __init__(self, n_clusters=4):
        super(GmmModel, self).__init__()
        self.n_clusters = n_clusters
        self.clustering = None
        self.regressors = [None for _ in range(n_clusters)]

    def partition(self, X, Y):
        labels = self.clustering.predict(X)
        partitions = []
        for l in range(self.n_clusters):
            label_indices = [i for i in range(X.shape[0]) if labels[i] == l]
            x_temp = np.zeros((len(label_indices), X.shape[1]))
            y_temp = np.zeros((len(label_indices),))

            for new_index, old_index in enumerate(label_indices):
                x_temp[new_index, :] = X[old_index, :]
                y_temp[new_index] = Y[old_index]

            partitions.append((x_temp, y_temp))
        return partitions

    # TODO: use weights?
    def _fit(self, X,Y, weights):
        #clustering = KMeans(n_clusters=N_CLUSTERS)
        #clustering = MeanShift()
        print('Training GMM')
        self.clustering = GMM(n_components=self.n_clusters) #min_covar=1)
        self.clustering.fit(X)

        partitions = self.partition(X, Y)
        for l in range(self.n_clusters):
            #self.regressors[l] = Ridge(alpha=10.0)
            #self.regressors[l] = SVR(kernel='poly', C=1e3, degree=2)
            # good for single cluster
            #self.regressors[l] = SVR(kernel='rbf', C=1e2, gamma=0.1)
            self.regressors[l] = SVR(kernel='rbf', C=1e3, gamma=0.1)
            #self.regressors[l] = RandomForestRegressor(n_estimators=10)
            x_temp, y_temp = partitions[l]
            MAX_SAMPLES = 10000
            actual_samples = min(x_temp.shape[0], MAX_SAMPLES)
            print('training for cluster %d (of size %d, but actual samples are %d)' % (l, x_temp.shape[0], actual_samples))

            rows_selected = np.random.randint(x_temp.shape[0],size=actual_samples)
            self.regressors[l].fit(x_temp[rows_selected,:], y_temp[rows_selected])


    def _predict(self, X):
        labels = self.clustering.predict(X)
        results = []
        for i in range(X.shape[0]):
            results.append(self.regressors[labels[i]].predict(X[i:(i+1),:]) )
        return np.hstack(results)

    def plot_clusters(self, X):
        if self.features:
            X = self.features(X)
        # plot
        plt.figure(figsize=(20,20))
        labels = self.clustering.predict(X)
        # TODO: this should come from the plant.
        LENGTH_1 = 1.0
        LENGTH_2 = 2.0
        colors = list('bgrcmyk')
        for l in range(self.n_clusters):
            x1_temp, y1_temp, x2_temp, y2_temp = [], [], [], []
            for i in range(X.shape[0]):
                if labels[i] == l:
                    x1 = LENGTH_1*np.sin(X[i,0])
                    y1 = -LENGTH_1*np.cos(X[i,0])

                    x2 = LENGTH_2*np.sin(X[i,2]) + x1
                    y2 = -LENGTH_2*np.cos(X[i,2]) + y1
                    noise_sigma = 0.04
                    x1_temp.append(x1 + np.random.normal(0,noise_sigma))
                    y1_temp.append(y1 + np.random.normal(0,noise_sigma))
                    x2_temp.append(x2 + np.random.normal(0,noise_sigma))
                    y2_temp.append(y2 + np.random.normal(0,noise_sigma))
            cur_color = colors[l % len(colors)]
            print('For cluster %s we go %d samples' % (cur_color, len(x1_temp),))
            plt.scatter(x1_temp, y1_temp, c=cur_color)
            plt.scatter(x2_temp, y2_temp, c=cur_color, marker='+')


class QuadraticModel(Model):
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.sklearn_model = Lasso()
        #self.sklearn_model = SVR(kernel='rbf', C=1e2, gamma=0.1)
    def quadratic_features(self, X):
        n, d = X.shape
        F = np.zeros((n, d*(d+1)/2 + d + 1))
        for i in range(X.shape[0]):
            x = X[i,:]
            F[i,0:1] = 1
            F[i,1:(d+1)] = x
            F[i, (d+1):(d*(d+1)/2+d+1)] = np.array([ x[j]*x[k] for j in range(d) for k in range(j,d)])
        return F

    def _fit(self, X,Y,weights):
        F = self.quadratic_features(X)
        self.sklearn_model.fit(F,Y)

    def _predict(self, X):
        F = self.quadratic_features(X)
        return self.sklearn_model.predict(F)
