import numpy as np
from math import *

#######A class representing different type of domains (sphere,ball,hypercube)
#### dim is the dimension, type \in {"sphere","ball","hypercube"} is the domain, param shows parameters of the domains
### param['radius'] is the radius for ball and sphere. param['len'] is the side length for hypercube
class Domain:


    def  __init__(self,dim,type,param):
        self.param, self.dim, self.type= param,int(dim), type


#### This function generate  uniform samples from the specific domains (sphere,ball,hypercube), the output is an
#### num by d array where each row is a sample of the domain.
    def generate_sample(self,num):
        res = np.zeros((num,self.dim))

        if(self.type == "hypercube"):
            res = np.random.uniform(0,self.param['len'],(num,self.dim))

        elif (self.type =="sphere"):
            temp = np.random.normal(0,1,(num,self.dim))
            temp = temp/np.linalg.norm(temp,axis=1)[:,np.newaxis]
            res = self.param['radius']*temp

        elif (self.type == "ball"):
            r = np.array([pow(r,1.0/self.dim) for r in np.random.uniform(0,1,num)])[:,np.newaxis]#*pow(self.param['radius'],3),1.0/3) for i in range(num)])[:,np.newaxis]
            temp = np.random.normal(0,1,(num,self.dim))
            temp = temp/np.linalg.norm(temp,axis=1)[:,np.newaxis]

            res = temp*r

        return res

###### A class representing different kernels: Gaussian,Polynomial
###### The variables: dim: shows dimension, type \in {"Gaussian","Polynomial"}: shows the type of kernel, domain is an instance of domain
##### param shows the prameter of kernel for Gaussian param['cov'] refers to its covariance matrix
### and for Polynomial param['ell'] shows the degree
class Kernel:

    def __init__ (self,type, dim, param, domain):
        self.type, self.dim , self.param , self.domain= type , dim , param, domain
        if(type == 'Gaussian'):
            self.invCov= np.linalg.inv(self.param['cov'])
            self.max_diagonal = 1
        if(type =='Polynomial'):
            self.ell = self.param['ell']
            if(self.domain.type =="ball" or self.domain.type =="sphere" ):
                self.max_diagonal= (1+self.domain.param['rad']**2)**self.param['ell']
            else:
                self.max_diagonal= ((1+self.dim*(self.domain.param['len']**2))**(self.param['ell']))

##### Return L(x,y) for the kernel L
    def get_value (self,x,y):
        if(self.type == "Gaussian"):
            return exp(-self.invCov.dot(x-y).dot(x-y))
        if(self.type == 'Polynomial'):
            return (1+x.dot(y))**int(self.ell)


######Implementation of the rejection sampler for the kernel. Points is the current point and n is their number of them. The function
#### returns a sample from the conditional probability distribution (CD_L(points,1) in the paper)
    def conditional_sampler(self,points,n,maxIter=500):

        ####### it first generate maxiter many uniform samples, if nonoe of them was accepted, repeats this process.
        proposal = self.domain.generate_sample(maxIter)
        test = np.random.uniform(0,1,maxIter)
        for i in range(maxIter):
            sample = proposal[i,:]
#####Computing the change in determinant after adding a new point uniformly sampled
            if(n != 0):
                cur_det = np.linalg.det(np.array(
                [[self.get_value(points[x,:],points[y,:]) for y in  range(n)] for x in range(n) ]))
                cur_point = np.append(points,sample[np.newaxis,:],axis=0)
                changed_det = np.linalg.det(np.array(
                [[self.get_value(cur_point[x,:],cur_point[y,:]) for y in range(n+1)] for x in range(n+1)]))
            else:
                cur_det=1
                changed_det = self.get_value(sample,sample)
####### checking the criteria for accepting a uniform sample
            if (self.type== 'Gaussian'):
                if (test[i] <= changed_det/cur_det):
                    return sample
            elif (test[i] <=changed_det/(cur_det*self.max_diagonal)):
                return sample


        return self.conditional_sampler(points,n,maxIter)




######## This objects defines a gibbs sampler, the paramters are $k$, and the kernel defining the chain. We run the chain in code for 5k^2 steps which seems
##### a good bound for empirical mixing.
class MCMC:



    def __init__(self,kernel,k):
        self.kernel, self.k = kernel,k
        self.NumOfIter=5*self.k*self.k
        self.max_sample= 200

    ######Finding the starting state
    def find_warm_start(self):
        start=np.zeros((self.k,self.kernel.dim))
        for i in range(self.k):
            start[i,:] = self.kernel.conditional_sampler(start[0:i,:],i,self.max_sample)

        return start

#######Running the chain while storing and reporting the whole trajectory to comptue PSRF values. Trajectory is a 3d array where the find_warm_start
###### coordinate correspond to the number of iterations, the second one determine the index of the point(1....k) and the last one determines a d-dimensional point
#####i.e. trajectory[n,i,j] is the value of the $j$-th coordinate of the  $i$-th point at step n.
    def output_trajectory (self,numofiter):
        k = self.k
        d = self.kernel.dim
        trajectory = np.zeros((numofiter,k,d))
        trajectory[0,:,:] = self.find_warm_start()
        for i in range(1,numofiter):
            index = np.random.randint(0,k)
            newpoint = self.kernel.conditional_sampler(trajectory[i-1,list(range(index))+list(range(index+1,k)),:],k-1,self.max_sample)
            trajectory[i,:,:] = np.append(trajectory[i-1,list(range(index))+list(range(index+1,k)),:],newpoint[np.newaxis,:],axis=0)
            np.random.shuffle(trajectory[i,:,:])

        return trajectory
######### The function runs the Gibbs chain and outputs a sample of the chain
    def run (self):
        cur_point = self.find_warm_start()
        for i in range(self.NumOfIter):
            index = np.random.randint(0,self.k)
            points_after_removal= cur_point[list(range(index))+list(range(index+1,self.k)), :]

            cur_point = np.append(points_after_removal, self.kernel.conditional_sampler(points_after_removal,self.k-1,self.max_sample)[np.newaxis,:],axis=0)

        return cur_point

if __name__ == '__main__':
    dim=30
    k=5
    rad=1
    sigmasq=0.25
    cov_matrix = sigmasq*np.eye(dim,dim)


    #####defines a sphere domain with radius "rad"
    domain= Domain(dim,"sphere", {'radius': rad})
    ###defines a guassian kernel with covariance matrix "cov_matrix" and a gibbs sampler with kernel "kernel" and number of points "k"
    kernel = Kernel('Gaussian',dim,{'cov': cov_matrix}, domain)
    mcmc= MCMC(kernel,k)
    sample =mcmc.run()
    print ("the generated sample is {}".format(sample))
