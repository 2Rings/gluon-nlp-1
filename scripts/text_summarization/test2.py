from gluonnlp.data import batchify as bf
import mxnet as mx
import numpy as np
a = ([1,2,3,4], 0)
b = ([5,7], 1)
c = ([1,2,3,4,5,6,7], 0)
d = mx.nd.array([[1,2,3,4], [5,2,7,8],[1,3,3,4], [5,3,7,8]], dtype=int)
# print d.shape
o = d[:,1]
print o
d1 = np.array([[1,2,3,4]])
u = mx.nd.arange(4,dtype = int)
print u
l = mx.nd.array(np.arange(0,4), dtype=int)
print l
m = mx.nd.array([1,2,3,4])
# print m.shape
data = np.random.uniform(0,1,10)
indices = mx.nd.stack(l,o,axis = 1)
# data = np.array([1,2,3,4])
# data = mx.nd.array(data)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x /e_x.sum()

data = softmax(data)
data = [[[data]]]
data = mx.nd.array(data)

print data
# indices = mx.nd.array(indices)
# print indices
gold_probs = mx.nd.gather_nd(data,indices)
print gold_probs
gold_probs = [gold_probs,gold_probs]
gold_probs = sum(gold_probs)
print gold_probs.mean()
# print l.dtype
e = np.array([[5,8, 7, 8],[1,2, 4,5]])
f = np.array([[5],[8],[ 7], [8]])
g = mx.nd.array([5,8, 7, 8])
h = [5,8, 7, 8]
i = [5,8, 7, 8]

# print d1,f
# print d1*f

# o1 = bf.Pad(axis = 1, pad_val = -1)([d,e])
# o2 = bf.Stack()([d,e])
# o3 = bf.Tuple(bf.Pad(), bf.Stack())([a,b])
# # o4 = bf.Pad()([a,b])
# # o4 = bf.Tuple(bf.Pad())([a,b])
# print o2
# print o3
# print o4

from gluonnlp.data import batchify as bf
import mxnet as mx
import numpy as np
a = [[[1,2], [4,5]],[[1,2], [4,5]],[[1,2], [4,5]]]
b = [[1,2], [4,5]]
a.append(b)
print a
c = ([1,2,3,4,5,6,7], 0)
d = mx.nd.array([[[1,2,3,4]], [[5,6,7,8]]])
d1 = np.array([[1,2,3,4]])

e = np.array([[5,8, 7, 8],[1,2, 4,5]])
f = np.array([[5],[8],[ 7], [8]])
g = mx.nd.array([5,8, 7, 8])
h = [5,8, 7, 8]
i = [5,8, 7, 8]

m = f.mean()
print m
def tt():
    print c
# print d1,f
# print d1*f

# o1 = bf.Pad(axis = 1, pad_val = -1)([d,e])
# o2 = bf.Stack()([d,e])
# o3 = bf.Tuple(bf.Pad(), bf.Stack())([a,b])
# # o4 = bf.Pad()([a,b])
# # o4 = bf.Tuple(bf.Pad())([a,b])
# print o2
# print o3
# print o4
