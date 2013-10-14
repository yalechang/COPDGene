"""
==============================================================================
HSIC measures dependence between two random variables.
This script implements Empirical HSIC, we have X=[x1,...,xm],Y=[y1,...,ym] be
a series of m independent observations drawn from p(x,y).An estimator of HSIC,
written HSIC(X,Y), is given by
HSIC(X,Y)=(m-1)^(-2)*tr(KHLH)
where:H,K,L belongs to R(mxm)
K(i,j) = k(xi,xj),which is the kernel function, we use Gaussian Kernel here
L(i,j) = l(yi,yj),which is the kernel function, we use Gaussian Kernel here
H(i,j) = delta(i,j)-1/m
==============================================================================
"""
#print __doc__

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def compute_kernel_matrix(x,y,sigma):
    """Compute Gaussian kernel of vector x and vector y

    Parameters
    ----------
    x : vector, shape(n_instances,1)    
    y : vecotr, shape(n_instances,1)
    sigma : scale of Gaussian kernel
    
    Returns
    ------
    s : kernel matrix
    """
    m = len(x)

    s = np.zeros((m,m))
    for i in range(len(x)):
        for j in range(i+1):
            s[i,j] = np.exp(-((x[i]-y[j])**2)/(2*sigma**2))
    for i in range(2,m):
        for j in range(0,i):
            s[i,j] = s[j,i]
    return s

def hsic(x,y,sigma):
    """Compute HSIC between two random variables

    Parameters
    ----------
    x : array, shape(n_instances,1)
        vector containing m observations of the first random variable
    y : array, shape(n_instances,1)
        vector containing m observations of the second random variable
    sigma : scale parameter for Gaussian kernel

    Returns
    -------
    hsic_value : float, HSIC value of the two input random variables
    """
    # m is the number of observations here
    m = len(x)
    gamma = 1.0/(2*sigma**2)

    k = rbf_kernel(x,x,gamma)
    l = rbf_kernel(y,y,gamma)
    for i in range(m):
        k[i,i] = 0
        l[i,i] = 0
    h = np.eye(m)-1.0/m
    hsic_value = (1.0/(m-1)**2)*np.trace(np.dot(np.dot(np.dot(k,h),l),h))
    return hsic_value

#Test module
if __name__ == "__main__":
    x = np.array([1,2,3]).reshape(3,1)
    y = np.array([13,17,39]).reshape(3,1)
    #y = np.random.rand(100,1)
    sigma = 10.0
    print(hsic(x,y,sigma))
