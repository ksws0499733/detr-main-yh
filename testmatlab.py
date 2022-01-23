import math
import numpy as np

c = np.array([[1,-1,0,0],
            [1,0,-1,0],
            [0,1,0,-1],
            [0,0,1,-1]])
d = np.array([[10],[10],[10],[10]])

D = np.matmul(d,d.T)
A = np.matmul(c,c.T)



print(D)
eigenvalue,featurevector=np.linalg.eig(A)
print('eigenvalue',eigenvalue)
print('featurevector',featurevector)