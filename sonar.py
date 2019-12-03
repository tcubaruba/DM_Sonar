import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA

# np.set_printoptions(threshold=sys.maxsize)


filename = 'runningexample.txt'

with open(filename) as f:
    data = f.readlines()
data = [x.strip() for x in data]  # split rows
data = [x.split() for x in data]  # split elemnets


# delete name of pint
y = []
for row in data:
    y.append(row[2])
    del row[2]

data = np.array(data)
y = np.array(y)

data = data.astype('float64')
y = y.astype('float64')

df = pd.DataFrame(data, columns=["x", "y"])
df["label"] = y

sns.pairplot(data=df, x_vars=["x"], y_vars=["y"], hue="label")
plt.show()


# Part 1: Apply EM algorithm

# print("True labels: ", y)

for k in range(1, 11):
    modelEM = GaussianMixture(n_components=k, max_iter=3, n_init=100, tol=0.00001, covariance_type='full')
    modelEM.fit(data)
    print("N components: ", k)
    print("Model score: ", modelEM.score(data, y))
    # print("Predicted labels:", modelEM.predict(data))


# Part 2 independent analysis and whilening

# covariance matrix
def calculateCovariance(X):
    meanX = np.mean(X, axis = 0)
    lenX = X.shape[0]
    X = X - meanX
    covariance = X.T.dot(X)/lenX
    return covariance


P = calculateCovariance(data)
print(P)
print(P.shape)

# eigenvalue decomposition
W, V = np.linalg.eig(P)
D = np.diag(W)

print(V@D@V.T - P)

# apply whitening ???

A = data.dot(V[1])
X = A/np.sqrt(W[1]+1e-5)

print(X)

X = X.reshape(-1, 1)

transformer = FastICA()
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)

