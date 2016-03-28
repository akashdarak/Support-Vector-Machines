import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse import *

from sklearn.datasets import load_svmlight_file
from scipy import sparse

X, y = load_svmlight_file("D:\Machine Learning\Assignment 3\scop_motif.data")

"""
#Normalized Kernel Matrix

from sklearn.preprocessing import normalize
Xnorm = normalize(X, norm='l2', axis=1)
X = Xnorm
"""

Xt = sp.sparse.csr_matrix.transpose(X)
kernel = X*Xt
K = kernel.toarray()
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(K, cmap= 'spectral')
plt.colorbar()
plt.title('Kernel Matrix')
plt.show() 
