import numpy as np


DAMPFACTOR = 10E-6


def dot(x,y):
    """
    Multiplication
    """
    z = np.zeros(x.shape)
    z[0,] = np.dot(x,y)
    z[1:,] = x[0,] * y[1:,] + y[0,] * x[1:,]
    return z


def identity(q):
    """
    Identity element
    """
    ii = np.zeros((q+1,))
    ii[0,] = 1.0
    return ii


def gen_psd(q):
    a = np.zeros((q+1,))
    x = np.random.randn(q,)
    t = np.linalg.norm(x) / np.random.rand()
    
    a[0] = t
    a[1:] = x
    
    return a


def gen_linearmap(d,q):
    A = np.zeros((d,q+1))
    for i in range(d):
        A[i,:] = gen_psd(q)
    return A


def inverse(x):
    """
    Compute inverse
    """
    lx = linear_representation(x)
    lxi = np.linalg.inv(lx)
    q, = x.shape
    q -= 1
    ii = identity(q)
    xinv = lxi @ ii
    return xinv


def squareroot(x):
    """
    Stable square-root
    """
    q, = x.shape
    t = x[0,]
    nn = np.linalg.norm(x[1:,])
    
    d = np.max([t**2-nn**2,0.0])
    c = t + np.sqrt(d)
    
    sqrtval = np.zeros(x.shape)
    sqrtval[0,] = np.sqrt(c/2)
    sqrtval[1:,] = x[1:,] * (1.0/np.sqrt(2*c))
    
    return sqrtval


# def squareroot(x):
#     """
#     Compute inverse
#     """
#     q, = x.shape
#     nn = np.linalg.norm(x[1:,])
    
#     if abs(nn) < 10E-10:
#         sqrtval = np.zeros(x.shape)
#         sqrtval[0,] = np.sqrt(x[0,])
        
#     else:
#         n = x[1:,] / np.linalg.norm(x[1:,])
#         c1 = np.zeros((q,)) # First eigenvector
#         c1[0,] = 0.5
#         c1[1:,] = 0.5 * n
#         c2 = np.zeros((q,)) # Second eigenvector
#         c2[0,] = 0.5
#         c2[1:,] = -0.5 * n
#         l1 = x[0,] + nn # First eigenvalue
#         l2 = x[0,] - nn # Second eigenvalue
        
#         sqrtval = np.sqrt(l1) * c1 + np.sqrt(l2) * c2
        
#     return sqrtval


def to_the_k(x,k):
    """
    Raise to the power k
    """
    q, = x.shape
    nn = np.linalg.norm(x[1:,])
    
    if abs(nn) < 10E-10:
        sqrtval = np.zeros(x.shape)
        sqrtval[0,] = x[0,]**k
        
    else:
        n = x[1:,] / np.linalg.norm(x[1:,])
        c1 = np.zeros((q,)) # First eigenvector
        c1[0,] = 0.5
        c1[1:,] = 0.5 * n
        c2 = np.zeros((q,)) # Second eigenvector
        c2[0,] = 0.5
        c2[1:,] = -0.5 * n
        l1 = x[0,] + nn # First eigenvalue
        l2 = x[0,] - nn # Second eigenvalue
        
        sqrtval = l1**k * c1 + l2**k * c2
        
    return sqrtval


def trace(x):
    """
    Trace
    """
    t = x[0,]
    return t


def linear_representation(x):
    """
    Linear representation
    """
    q, = x.shape
    lx = np.zeros((q,q))
    lx[0,0] = x[0,]
    lx[1:,1:] = x[0,] * np.identity(q-1)
    lx[1:,0] = x[1:,]
    lx[0,1:] = x[1:,].T
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
    """
    Compute quadratic representation
    """
    lxy = linear_representation(x) @ linear_representation(y)
    lyx = linear_representation(y) @ linear_representation(x)
    qxy = lxy + lyx - linear_representation(dot(x,y))
    return qxy



def scaling(x,y,dampfactor=DAMPFACTOR):
    ee = identity(x.shape[0]-1)
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