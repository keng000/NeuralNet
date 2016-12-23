import numpy as np

class Perceptron(object):
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

class StochasticGradientDescent(object):
    def __init__(self, eta=0.0001, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.weight_init = False

        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._init_weight(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weight(xi, target))

            avg_cost = sum(cost)/len(y)

            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        if not self.weight_init:
            self._init_weight(X.shape[1])

        if y.ravel.shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weight(xi, target)
        else:
            self._update_weight(X, y)

        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _init_weight(self, m):
        self.w_ = np.zeros(m + 1)
        self.weight_init = True

    def _update_weight(self, xi, target):
        output = self.net_inpiut(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error

        cost = error*error
        return cost

    def net_inpiut(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(net_input(X) >= 0, 1, -1)
