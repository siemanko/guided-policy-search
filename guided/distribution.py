import numpy as np
from numpy import dot


from numpy.random import multivariate_normal
from numpy.linalg import inv,norm

class IW(object):
    def __init__(self, size):
        self.size = size
        self.phi = np.identity(self.size)
        self.nu = self.size

    def include_samples(self, samples, mu=None):
        if mu is None:
           mu = np.zeros((1,self.size))
        n = len(samples)
        X = np.zeros((n,self.size))
        for i in range(n):
            X[i,:] = samples[i] - mu
        A = dot(X.T, X)
        self.phi = self.phi + A
        self.nu = self.nu + n

    def mean(self):
        return self.phi / (self.nu - self.size - 1)

def IWDemo():
    mean = np.array([-23,23,10])
    var = np.array([[3,2,1],[2,3,2],[1,2,3]])
    def sample():
        return multivariate_normal(mean,var)

    print(norm(dist.main() - var))

    dist = IW(3)
    for _ in range(10):
        dist.include_samples([sample() for _ in range(10)], mean)

    print(norm(dist.mean() - var))
