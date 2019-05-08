import numpy as np
import math
from Domain import  *



##### Computes a 1D array of size d, where coordinate i, computes the psrf value of the trajectory of the walk projected at coordinate i
#### The input is a 4D array of the trajectory as output by the trajectory method of MCMC class

def PSRF(samples):
    n,m,k,d = samples.shape
    V= np.zeros((d,k,k))
    psrf = np.zeros(d)
    for i in range(d):
        fixed_coor = samples[:,:,:,i]
        totalmean = np.mean(fixed_coor, axis=(0,1))
        in_group_mean = np.mean(fixed_coor,axis=0)
        var_between = in_group_mean.T.dot(in_group_mean)-m*totalmean[:,np.newaxis].dot(totalmean[np.newaxis,:])
        var_between = var_between/(m-1)
        var_inside = np.zeros((k,k))

        for j in range(m):
            fixed_trajectory = fixed_coor[:,j,:]
            mean = np.mean(fixed_trajectory,axis=0)
            var =  fixed_trajectory.T.dot(fixed_trajectory)-n*mean[:,np.newaxis].dot(mean[np.newaxis,:])
            var = var/(n-1)
            var_inside = var_inside+ var

        var_inside = var_inside/m
        V[i,:,:] = (n-1.0)*var_inside/n+(1+1.0/m)*var_between
        psrf[i] = np.linalg.norm(np.linalg.inv(var_inside).dot(V[i,:,:]),2)

    return psrf




#### NOTE:The array of PSRF values over time is not sorted, so theoretically binary search is not the correct way to find the first
#### index where $\alpha \leq 1.1$, however, first it is an upperbound on mixing. and secondly, we also tried the result
### with iterating over all indices and find the first place \alpha <1.1. But the changes were negligible. We chose this way for performance issues.

##### This simple functions, uses the PSRF function calls to find empirical mixing time, which in our case is the first time  the average PSRF  drops below 1.1
#### the input is a 4D array as output by the trajectory function of class MCMC
def Mixing(samples):
    low = 0
    high = samples.shape[0]

    while low < high-1:
        l = int((low+high)/2)
        psrf = PSRF(samples[0:l,:,:,:])
        mean = np.mean(psrf)
        if(mean >1.1):
            low = l
        else:
            high = l


    return high


###### A sample code that computes the empirical mixing for a polynomial kernel with ell=5 and dim=10 and k=5
if __name__ == '__main__':
    dim=10
    length=1
    ell=5
    m = 10
    k=5
    n = 100
    domain= Domain(dim,"hypercube", {'len': length})
    kernel = Kernel('Polynomial',dim,{'ell' : ell},domain)
    trajectory = np.zeros((n,m,k,dim))
    mcmc= MCMC(kernel,k)
    for  i in range(m):
        trajectory[:,i,:,:]= mcmc.output_trajectory(n)

    psrf = PSRF(trajectory)
    print ("the PSRF for the trajectory is {}".format(psrf))
    tau = Mixing(trajectory)
    print ("the empirical mixing time  is {}".format(tau) )
