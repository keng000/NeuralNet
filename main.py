# -*- coding: utf-8 -*-
import NeuralNet as NN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

'''
# irisデータの描画
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0],X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.legend(loc = 'upper left')
plt.ylabel('Petal lenght[cm]')
plt.xlabel('Sepal length[cm]')
plt.show()
'''

'''
# fiting過程における誤識別数の経過
nn = NN.Perceptron()
nn.fit(X, y)

#plt.plot(range(1, len(nn.error)+1), nn.error, color='red', marker='o')
#plt.show()
print nn.w_
'''

'''
#決定境界の描画
from matplotlib.colors import ListedColormap
def plot_color_map(X, y, classifier, resolution=0.01):
    markers = ['s', 'o', 'x', '^', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() +1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() +1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
nn = NN.Perceptron()
nn.fit(X, y)
plot_color_map(X, y, classifier=nn)
plt.legend(loc = 'upper left')
plt.ylabel('Petal lenght[cm]')
plt.xlabel('Sepal length[cm]')
plt.show()
'''


#　adaptive linear neuron
nn1 = NN.ADALINE(eta=0.0004)
nn2 = NN.ADALINE(eta=0.0004)

nn1.fit_at_once(X, y)
nn2.fit_each_sample(X, y)

fid, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ax[0].plot(range(1, len(nn1.cost_) + 1), nn1.cost_, marker='o')
ax[0].set_title("fit at once eta=0.0004")
ax[1].plot(range(1, len(nn2.cost_) + 1), nn2.cost_, marker='o')
ax[1].set_title("fit each sample eta=0.0004")
plt.show()
