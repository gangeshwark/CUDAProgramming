import numpy as np
import cudamat as cm
import time

st = time.time()
cm.cublas_init()

# create two random matrices and copy them to the GPU
a = cm.CUDAMatrix(np.random.rand(32, 256))
b = cm.CUDAMatrix(np.random.rand(256, 32))

# perform calculations on the GPU
d = cm.dot(a, b)
d = d.sum(axis = 0)
print (d.transpose()).shape, d.asarray().T
print "{0} secs".format(time.time()-st)

# copy d back to the host (CPU) and print
#print(d.asarray())
st = time.time()
c = np.dot(np.random.rand(32, 256),np.random.rand(256, 32))
print c

print "{0} secs".format(time.time()-st)