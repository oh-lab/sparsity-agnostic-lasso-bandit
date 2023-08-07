import numpy as np
import random
from models import LassoBandit, DRLassoBandit, SALassoBandit
import time
import argparse

parser = argparse.ArgumentParser(description='sparse bandit')

parser.add_argument('--N', type = int, default=20, help='number of arms')
parser.add_argument('--T', type = int, default=1000, help='horizon')
parser.add_argument('--d', type = int, default=100, help='feature dimension')
parser.add_argument('--s0', type = int, default=5, help='sparsity')
parser.add_argument('--rho_sq', type = float, default=0.3, help='correlation (used for gaussian)')
parser.add_argument('--R', type = float, default=0.5, help='noise')
parser.add_argument('--dist', type = int, default=0, help='context distribution - 0:gaussian, 1:uniform, 2:elliptical')
parser.add_argument('--id', type = int, default=0, help='job ID')


def sample_spherical(N, k):
    vec = np.random.randn(k, N)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_elliptical(N, d, k, mu):
    S = sample_spherical(N, k)
    A = np.random.rand(d,k)
    R = np.random.normal(size=N)
    return mu + A.dot(R*S)


def main():
	args = parser.parse_args()
	random.seed(args.id)

	N = args.N
	d = args.d
	s0 = args.s0
	R = args.R
	T = args.T

	sigma_sq=1.
	rho_sq= args.rho_sq
	V=(sigma_sq-rho_sq)*np.eye(N)+rho_sq*np.ones((N,N))

	beta=np.zeros(d)
	inds=np.random.choice(range(d),s0,replace=False)
	beta[inds]=np.random.uniform(0.,1.,s0)

	savename = "results/sparseBandit_N={}_d={}_s={}_rho={}_dist={}_id={}.csv".format(N, d, s0, rho_sq, args.dist, args.id)

	simul_n=1

	cumulated_regret_Lasso=[]
	cumulated_regret_DR=[]
	cumulated_regret_SALasso=[]

	for simul in range(simul_n):

		M1=LassoBandit(q=1+int(0.001*(s0**2)*np.log(d)),h=5,lam1=0.2/s0,lam2=0.5,d=d,N=N)
		# M2=DRLassoBandit(lam1=0.1*s0,lam2=0.5,d=d,N=N,tc=1,tr=True,zt=0.1*s0*np.log(d))
		# M1=LassoBandit(q=1,h=5,lam1=0.05,lam2=0.05,d=d,N=N)
		M2=DRLassoBandit(lam1=1.,lam2=0.5,d=d,N=N,tc=1,tr=True,zt=10)
		M3=SALassoBandit(sigma=0.2,d=d,N=N)

		RWD1=list()
		RWD2=list()
		RWD3=list()
		optRWD=list()

		for t in range(T):
			if args.dist == 0:
				x=np.random.multivariate_normal(np.zeros(N),V,d).T
			elif args.dist == 1:
				x=(np.random.random((N, d)) * 2 - 1)
			elif args.dist == 2:
				x=sample_elliptical(N, d, int(d/2), 0).T

			x_stack=x.reshape(N*d)

			err=R*np.random.randn()

			a1=M1.choose_a(t+1,x_stack)
			rwd1=np.dot(x[a1],beta)+err
			RWD1.append(np.dot(x[a1],beta))
			M1.update_beta(rwd1,t+1)

			a2=M2.choose_a(t+1,x)
			rwd2=np.dot(x[a2],beta)+err
			RWD2.append(np.dot(x[a2],beta))
			M2.update_beta(rwd2,t+1)

			a3=M3.choose_a(t+1,x)
			rwd3=np.dot(x[a3],beta)+err
			RWD3.append(np.dot(x[a3],beta))
			M3.update_beta(rwd3,t+1)

			optRWD.append(np.amax(np.dot(x,beta)))
	    
		cumulated_regret_Lasso.append(np.cumsum(optRWD)-np.cumsum(RWD1))
		cumulated_regret_DR.append(np.cumsum(optRWD)-np.cumsum(RWD2))
		cumulated_regret_SALasso.append(np.cumsum(optRWD)-np.cumsum(RWD3))

	regret = np.vstack([np.asarray(cumulated_regret_Lasso), np.asarray(cumulated_regret_DR), np.asarray(cumulated_regret_SALasso)])
	np.savetxt(savename, regret, delimiter=",")
	
if __name__ == '__main__':
	main()
