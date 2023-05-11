import numpy as np


DAMPFACTOR = 10E-6


def dot(x,y):
    """
    Multiplication
    """
    z = np.multiply(x,y)
    return z


def identity(q):
    """
    Identity element
    """
    ii = np.ones((q,))
    return ii


def gen_psd(q):
    x = np.random.rand(q)
    return x


def gen_linearmap(d,q):
    A = np.zeros((d,q))
    for i in range(d):
        A[i,:] = gen_psd(q)
    return A


def inverse(x):
    """
    Compute inverse
    """
    y = np.zeros(x.shape)
    q, = x.shape
    for i in range(q):
        y[i,] = 1.0 / x[i,]
    return y


def squareroot(x):
    """
    Compute inverse
    """
    y = np.zeros(x.shape)
    q, = x.shape
    for i in range(q):
        y[i,] = np.sqrt(x[i,])
    return y


def trace(x):
    """
    Trace
    """
    t = np.sum(x)
    return t


def linear_representation(x):
    """
    Linear representation
    """
    lx = np.diag(x)
    return lx


def quadratic_representation(x):
    """
    Compute inverse
    """
    x2 = dot(x,x)
    l2x = linear_representation(x) @ linear_representation(x)
    lx2 = linear_representation(x2)
    qx = 2 * l2x - lx2
    return qx


def scaling(x,y,dampfactor=DAMPFACTOR):
    ee = identity(x.shape[0])
    x_sqrt = squareroot(x+ee*dampfactor)
    #print(x_sqrt)
    qx = quadratic_representation(x_sqrt)
    #print(qx)
    qxy2 = squareroot(qx @ y + ee*dampfactor)
    #print(qxy2)
    E = np.identity(qx.shape[0])
    qxinv = np.linalg.inv(qx + E * dampfactor)
    #qxinv = np.linalg.inv(qx + E * 10E-10) # Original un-damped
    #print(qxinv)
    w = qxinv @ qxy2
    return w


def to_the_k(x,k):
    y = x.copy()
    d = y.shape[0]
    for i in range(d):
        y[i,] = y[i,]**k
    return y