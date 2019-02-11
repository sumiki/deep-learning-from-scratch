

# 3.4.2 ３層ニューラルネットワークの計算

# 2=>3=>2 のニューラルネットワーク

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print('------1層目------------------------------------')

X = np.array([1.0, 0.5])
W1 = np.array( [[ 0.1, 0.3, 0.5 ], [ 0.2, 0.4, 0.6 ]] ) # 2,3format
B1 = np.array( [0.1, 0.2, 0.3] )

print( W1.shape )
print( X.shape )
print( B1.shape )

A1 = np.dot(X, W1) + B1

print(A1)
Z1 = sigmoid(A1)
print(Z1)


print('------2層目------------------------------------')


W2 = np.array( [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]] ) # 3,2 format
B2 = np.array( [0.1, 0.2] )

A2 = np.dot( Z1, W2 ) + B2
print(A2)
Z2 = sigmoid(A2)

print( Z2 )

print('------3層目------------------------------------')


W3 = np.array( [ [0.1, 0.3], [0.2, 0.4] ] )
B3 = [0.1,0.2]

A3 = np.dot(Z2, W3) + B3
print(A3)
Z3 = sigmoid(A3)
print(Z3)