import numpy as np
import matplotlib.pyplot as plt

X = ['A','B','C']
Y = [1,1,1]
Z = [0.5,5,6]

def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge")   
    plt.xticks(_X, X)
    
subcategorybar(X, [Y,Z,Y])

plt.show()