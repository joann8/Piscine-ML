import numpy as numpy
from zscore import zscore

X = numpy.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(X))


Y = numpy.array([2, 14, -13, 5, 12, 4, -19])
print(zscore(Y))
