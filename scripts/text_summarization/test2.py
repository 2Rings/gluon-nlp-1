from gluonnlp.data import batchify as bf
import mxnet as mx
import numpy as np
a = ([1,2,3,4], 0)
b = ([5,7], 1)
c = ([1,2,3,4,5,6,7], 0)
d = mx.nd.array([[[1,2,3,4]], [[5,6,7,8]]])
d1 = np.array([[1,2,3,4]])

e = np.array([[5,8, 7, 8],[1,2, 4,5]])
f = np.array([[5],[8],[ 7], [8]])
g = mx.nd.array([5,8, 7, 8])
h = [5,8, 7, 8]
i = [5,8, 7, 8]

print d1,f
print d1*f

# o1 = bf.Pad(axis = 1, pad_val = -1)([d,e])
# o2 = bf.Stack()([d,e])
# o3 = bf.Tuple(bf.Pad(), bf.Stack())([a,b])
# # o4 = bf.Pad()([a,b])
# # o4 = bf.Tuple(bf.Pad())([a,b])
# print o2
# print o3
# print o4
