'''
Created on Oct 25, 2010

@author: arnaud
'''

from numpy import ones, zeros, dot, tanh, exp, linalg, random, sqrt
from scipy.io import savemat, loadmat



class MultilayerPerceptron:
    def __init__(self,T,bias=True,linOutput=True):
        if bias == True:
            T[0] = T[0]+1   #bias
        self.T = list(T)
        self.H = [ones((n,1)) for n in self.T]
        self.S = [zeros((n,1)) for n in self.T]
        self.M = [(-3*ones((self.T[d],self.T[d+1])).T + 6*random.rand(self.T[d],self.T[d+1]).T)/sqrt(max(T[d],T[d+1])) for d in range(0,len(self.T)-1)]
        self.E = list(self.H)
        self.DM = list(self.M)
        self.pDM = list(self.DM)
        self.beta = 1.
        self.linOutput = linOutput

        
    def process(self,X):
        self.H[0][0:len(X)] = X     #bias
        self.S[0] = self.H[0]
        for l in range(0,len(self.T)-1):
            self.H[l+1] = dot(self.M[l],self.S[l])
            self.S[l+1] = self.sig(self.H[l+1],self.beta)
        if self.linOutput == True:
            self.S[len(self.T)-1] = self.H[len(self.T)-1]
        #self.showproc()
        return self.S[len(self.T)-1]
            
    def learn(self,X,Y, alpha = 0.01, momentum = 0.):
        self.process(X)
        m = len(self.T)-1
        if self.linOutput == False:
            self.E[m] = self.dersig(self.H[m],self.beta)*(Y-self.S[m])
        else:
            self.E[m] = Y-self.S[m]
        for l in range(m,0,-1):
            self.E[l-1] = self.dersig(self.H[l-1],self.beta)*dot(self.M[l-1].T,self.E[l])
            self.DM[l-1] = alpha*dot(self.E[l],self.S[l-1].T)
            self.M[l-1] = self.M[l-1] + self.DM[l-1] + momentum*self.pDM[l-1]
        self.pDM = list(self.DM)
        #self.showlearn()
        
        
    def sig(self,x,beta):
        return tanh(beta*x)
        return 1./(1.+exp(-beta*x))
    
    
    def dersig(self,x,beta):
        return beta*(1.-self.sig(x,beta)**2)
        return beta*(1.-self.sig(x,beta))*self.sig(x,beta)
    
    def save(self,fname):
        dict = {}
        basename = "../res/"
        for i in range(0,len(self.M)):
            dict.update({'m' + str(i): self.M[i]})
        savemat(basename + fname, dict)
        
    def load(self,fname):
        basename = "../res/"
        mweights = loadmat(basename + fname +'.mat')
        for i in range(0,len(self.M)):
            name = 'm' + str(i)
            self.M[i] = mweights[name]
        
    def wNorm(self):
        norm = []
        for m in self.M:
            norm.append(linalg.norm(m))
        return norm
    
    def bumpWeights(self,delta):
        for m in self.M:
            self.M[self.M.index(m)] = m + -delta*ones(m.shape) + 2*delta*random.rand(m.shape[0],m.shape[1])

            
        
#    def showproc(self):
#        print 'Process \n'
#        for k in range(0,len(self.T)-1):
#            print self.M[k], 'M[', k, ']', '\n'
#            print self.S[k], 'S[', k, ']', '\n'
#            print self.H[k+1], 'H[', k+1, ']', '\n'
#            print self.S[k+1], 'S[', k+1, ']', '\n'
#
#    def showlearn(self):
#        print 'Error Back-propagation \n'
#        for k in range(0,len(self.T)-1):
#            print self.E[k], 'E[', k, ']', '\n'        
#            print self.DM[k], 'DM[', k, ']', '\n'
#        print self.E[len(self.T)-1]
