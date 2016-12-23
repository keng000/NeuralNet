# -*- coding: utf-8 -*-
import NeuralNet as NN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def divide_into_train_and_test(X, y, n_train):
    r = np.random.permutation(len(y))
    X_ = X[r]
    y_ = y[r]
    return X_[:n_train], y_[:n_train], X_[n_train:], y_[n_train:]

df = pd.read_csv('iris.csv')
LABEL = df.iloc[50:, 4].values
LABEL = np.where(LABEL == 'Iris-versicolor', -1, 1)
FEAT = df.iloc[50:, [0,3]].values

X_train, y_train, X_test, y_test = divide_into_train_and_test(FEAT, LABEL, 90)

nn = NN.Perceptron()

nn.fit(X_train, y_train)
#plt.plot(range(1, len(nn.error)+1), nn.error, marker='o', color='red')
plot_color_map(X_train, y_train, classifier=nn)
plt.show()
result = nn.predict(X_test)
per = np.where(result == y_test, 1, 0)
print "%.1f%%"%(float(per.sum())*100/len(per))

dx1 = X_train[y_train == -1]
dx2 = X_train[y_train == 1]
dx3 = X_test[y_test == -1]
dx4 = X_test[y_test == 1]
plt.scatter(dx1[:,0], dx1[:,1], marker='o', color='red', label='versicolor')
plt.scatter(dx2[:,0], dx2[:,1], marker='o', color='blue' , label='virginica')
plt.scatter(dx3[:,0], dx3[:,1], marker='o', color='green', label='versicolor_pre')
plt.scatter(dx4[:,0], dx4[:,1], marker='o', color='cyan' , label='virginica_pre')
plt.legend(loc = 'upper left')
plt.show()
