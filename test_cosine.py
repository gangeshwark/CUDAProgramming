# Benchmarking cosine similarity of scipy on CPU and mine on GPU with CUDA Cores

import timeit
import numpy as np
import cudamat as cm
import time

from pycuda import driver
from pycuda import gpuarray
from scipy import spatial
import skcuda.linalg as linalg
import skcuda.misc as misc


def skcuda_linalg(a, b):
    linalg.init()
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = linalg.dot(a_gpu, b_gpu, 'T')
    a_nrm = linalg.norm(a_gpu)
    b_nrm = linalg.norm(b_gpu)
    type(a_nrm)
    ans = misc.divide(c_gpu, a_nrm* b_nrm)
    print ans

def sp_cosine_similarity(a, b):
    print 1 - spatial.distance.cosine(a, b)


def cosine_similarity_gpu(a, b):
    st = time.time()
    a = cm.CUDAMatrix(a)
    b = cm.CUDAMatrix(b)
    print time.time() - st
    mul = cm.dot(a.transpose(), b)
    print time.time() - st
    _a = a.euclid_norm()
    _b = b.euclid_norm()
    print time.time() - st
    d = mul.divide(_a * _b)
    print d.asarray()
    print time.time() - st


def test():

    algorithms = (sp_cosine_similarity, cosine_similarity_gpu, skcuda_linalg)
    val = np.random.rand(100000, 1)
    vector = np.array(val.tolist())
    matrix = np.array(val.tolist())
    vector = np.random.rand(100000, 1)
    matrix = np.random.rand(100000, 1)
    for algorithm in algorithms:
        t = timeit.timeit(lambda: algorithm(matrix, vector), number=1)
        print("{:s} {:f}".format(algorithm.__name__, t))


if __name__ == '__main__':
    driver.init()
    ngpus = driver.Device.count()
    ctx = driver.Device(0).make_context()
    dev = ctx.get_device()
    test()
    ctx.pop()
