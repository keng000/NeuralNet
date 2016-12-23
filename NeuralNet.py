import numpy as np

class NeuralNet(object):
    def __init__(self, n_iter=10, eta=0.001):
        self.n_iter = n_iter
        self.eta = eta

    def fit(self,X,y):
        self.w_ = np.zeros(X.shape[1]+1)
        self.error = []

        for epoc_time in range(self.n_iter):
            error = 0
            for target, xi in zip(y, X):
                predict = self.predict(xi)
                update = self.eta * (target - predict)
                self.w_[1:] += update * xi
                self.w_[0] += update

                if update != 0.0:
                    error += 1
            self.error.append(error)

    def net_input(self,xi):
        return np.dot(xi,self.w_[1:]) + self.w_[0]

    def predict(self,xi):
        return np.where(self.net_input(xi) >= 0, 1, -1)

class ADALINE(object):
    def __init__(self, eta=0.0003, n_iter=50):
        self.eta = eta
        self.n_iter = 50

    def fit_at_once(self, X, y):
        '''
        X : n_samples x p_characters
        y : n_samples x 1
        w_ : p_characters x 1
        output : n_samples X 1 <- (sigma)w_*xi
        '''

        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (errors*errors).sum()
            self.cost_.append(cost)
        return self

    def fit_each_sample(self, X, y):
        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []

        for i in range(self.n_iter):
            for i,xi in enumerate(X):
                output = self.net_input(xi)
                error = y[i] - output
                self.w_[1:] += self.eta * error * xi
                self.w_[0] += self.eta * error

            output = self.net_input(X)
            errors = (y - output)
            cost = (errors*errors).sum()
            self.cost_.append(cost)
        return self

    def net_input(self, xi):
        return np.dot(xi, self.w_[1:]) + self.w_[0]

    def activation(self, xi):
        return self.net_input(xi)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)
