import numpy as np
from scipy.stats import norm
from sklearn import linear_model

# Lasso Bandit by Bastani and Bayati (2015). Online decision-making with high-dimensional covariates.
class LassoBandit:
    def __init__(self,q,h,lam1,lam2,d,N):
        self.Tx=np.empty((N, 0)).tolist()
        self.Sx=np.empty((N, 0)).tolist()
        self.Tr=np.empty((N, 0)).tolist()
        self.Sr=np.empty((N, 0)).tolist()
        self.q=q
        self.h=h
        self.lam1=lam1
        self.lam2=lam2
        self.d=d
        self.N=N
        self.beta_t=np.zeros((N,N*d))
        self.beta_a=np.zeros((N,N*d))
        self.n=0
        self.lasso_t=linear_model.Lasso(alpha=self.lam1) #for force-sample estimator
    
    def choose_a(self,t,x): #x is N*d-dim vector 
        if t==((2**self.n-1)*self.N*self.q+1):
            self.set=np.arange(t,t+self.q*self.N)
            self.n+=1
        if t in self.set:
            ind=list(self.set).index(t)
            self.action=ind//self.q
            self.Tx[self.action].append(x)
        else:
            est=np.dot(self.beta_t,x) #N by 1
            max_est=np.amax(est)
            self.K=np.argwhere(est>max_est-self.h/2.) # action indexes
            est2=[np.dot(x,self.beta_a[k[0]]) for k in self.K]
            self.action=self.K[np.argmax(est2)][0]
        self.Sx[self.action].append(x)
        return(self.action)            
             
    def update_beta(self,rwd,t):
        if t in self.set:
            self.Tr[self.action].append(rwd)
            self.lasso_t.fit(self.Tx[self.action],self.Tr[self.action])
            self.beta_t[self.action]=self.lasso_t.coef_
        self.Sr[self.action].append(rwd)
        lam2_t=self.lam2*np.sqrt((np.log(t)+np.log(self.N*self.d))/t)
        lasso_a=linear_model.Lasso(alpha=lam2_t)
        if t>5:
            lasso_a.fit(self.Sx[self.action],self.Sr[self.action])
            self.beta_a[self.action]=lasso_a.coef_
        
        

# DR Lasso Bandit by Kim and Paik (2019). Doubly-Robust Lasso Bandit.
class DRLassoBandit:
    def __init__(self,lam1,lam2,d,N,tc,tr,zt):
        self.x=[]
        self.r=[]
        self.lam1=lam1
        self.lam2=lam2
        self.d=d
        self.N=N
        self.beta=np.zeros(d)
        self.tc=tc
        self.tr=tr
        self.zt=zt
        
    def choose_a(self,t,x):  # x is N*d matrix
        if t<self.zt:
            self.action=np.random.choice(range(self.N))
            self.pi=1./self.N
        else:
            uniformp=self.lam1*np.sqrt((np.log(t)+np.log(self.d))/t)
            uniformp=np.minimum(1.0,np.maximum(0.,uniformp))
            choice=np.random.choice([0,1],p=[1.-uniformp,uniformp])
            est=np.dot(x,self.beta)
            if choice==1:
                self.action=np.random.choice(range(self.N))
                if self.action==np.argmax(est):
                    self.pi=uniformp/self.N+(1.-uniformp)
                else:
                    self.pi=uniformp/self.N            
            else:
                self.action=np.argmax(est)
                self.pi=uniformp/self.N+(1.-uniformp)
        self.x.append(np.mean(x,axis=0))
        self.rhat=np.dot(x,self.beta)
        return(self.action)            
             
     
    def update_beta(self,rwd,t):
        pseudo_r=np.mean(self.rhat)+(rwd-self.rhat[self.action])/self.pi/self.N
        if self.tr==True:
            pseudo_r=np.minimum(3.,np.maximum(-3.,pseudo_r))
        self.r.append(pseudo_r)
        if t>5:
            if t>self.tc:
                lam2_t=self.lam2*np.sqrt((np.log(t)+np.log(self.d))/t) 
            lasso=linear_model.Lasso(alpha=lam2_t)
            lasso.fit(self.x,self.r)
            self.beta=lasso.coef_
            

# Sparsity-Agnostic Lasso Bandit by Oh, Iyengar, Zeevi (2021)
class SALassoBandit:
    def __init__(self,sigma,d,N):
        self.x=[]
        self.r=[]
        self.sigma=sigma
        self.d=d
        self.N=N
        self.beta=np.zeros(d)
        
    def choose_a(self,t,x): 
        est=np.dot(x,self.beta)
        self.action=np.argmax(est)
        self.x.append(x[self.action])
        return(self.action)            
             
     
    def update_beta(self,rwd,t):
        self.r.append(rwd)
        if t>5:
            lam_t=2*self.sigma*np.sqrt((4*np.log(t)+2*np.log(self.d))/t) 
            lasso=linear_model.Lasso(alpha=lam_t)
            lasso.fit(self.x,self.r)
            self.beta=lasso.coef_