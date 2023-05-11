import numpy as np
from scipy.linalg import sqrtm


DAMPFACTOR = 10E-6


def _vectorize(X):
    q1,q2 = X.shape
    return np.reshape(X,(q1*q2,))


def _matricize(X):
    q, = X.shape
    q = int(np.round(np.sqrt(q)))
    return np.reshape(X,(q,q))


def dot(x,y):
    """
    Multiplication
    """
    x = _matricize(x)
    y = _matricize(y)
    z = (x@y + y@x) / 2
    z = _vectorize(z)
    return z


def identity(q):
    """
    Identity element
    """
    ii = np.identity(q)
    ii = _vectorize(ii)
    return ii


def gen_psd(q):
    X = np.random.randn(q,q)
    X = X@X.T
    X = _vectorize(X)
    return X


def gen_linearmap(d,q):
    A = np.zeros((d,q*q))
    for i in range(d):
        A[i,:] = gen_psd(q)
    return A


def inverse(x):
    """
    Compute inverse
    """
    x = _matricize(x)
    x = np.linalg.inv(x)
    x = _vectorize(x)
    return x


def squareroot(x):
    """
    Compute inverse
    """
    x = _matricize(x)
    x = sqrtm(x)
    x = _vectorize(x)
    return x


def to_the_k(x,k):
    x = _matricize(x)
    u,v = np.linalg.eig(x)
    d = u.shape[0]
    for i in range(d):
        u[i,] = u[i,]**k
    x = (v @ np.diag(u)) @ v.T
    x = _vectorize(x)
    return x


def trace(x):
    """
    Trace
    """
    x = _matricize(x)
    t = np.trace(x)
    return t


def linear_representation(x):
    """
    Linear representation
    """
    x = _matricize(x)
    q,_ = x.shape
    ii = np.identity(q)
    lx = (np.kron(ii,x) + np.kron(x,ii))/2
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


def quadratic_representation2(x,y):
    lxy = linear_representation(x) @ linear_representation(y)
    lyx = linear_representation(y) @ linear_representation(x)
    qxy = lxy + lyx - linear_representation(dot(x,y))
    return qxy


def scaling(x,y,dampfactor=DAMPFACTOR):
    ee = identity(int(np.sqrt(x.shape[0])))
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