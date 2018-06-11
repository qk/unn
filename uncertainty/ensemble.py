import os
import numpy as np
from uncertainty.base import UncertaintyEstimator, PipeMixin
import multiprocessing as mp
import time
from utils import Frame
import scipy
import scipy.stats

__all__ = [ 'EnsembleUncertainty' ]

def _fit(network, name, va):
	tr = Frame(np.load(os.path.join("trained_models", name, "tr.npz")))
	va = Frame(va)
	est = network(name=name)
	est.fit(tr, va)
	pr = est.predict(va)
	rmse = ((va["T"] - pr["Y"])**2).mean()**0.5
	print("network", name, "trained with", rmse, "validation rmse")
	return rmse

def _predict(network, name, data):
	data = Frame(data)
	est = network(name=name)
	pr = est.predict(data)
	return pr["Y"]

def _assign_gpu(lock, gpus, f, args):
	# not very elegant, but at least somewhat concise
	gpu = None
	while gpu is None:
		with lock:
			for i,n in enumerate(gpus):
				if n > 0:
					gpus[i] -= 1
					gpu = i
					break
		if gpu == None: # should be prevented by limiting poolsize to sum(gpus)
			print("ERROR, gpu is None with", gpus)
			time.sleep(60)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else '0'

	result = f(*args)

	with lock:
		gpus[gpu] += 1
	return result


class EnsembleUncertainty(UncertaintyEstimator):
	# bootstraps m datasets from D of size equal to D
	def __init__(self, network, m=4, method="pairs", balancing=False, k=0, folder="", gpus=(1,)):
		'''
		params:
		network -- a parameterless network constructor
		method -- "pairs" (default) or "residuals", decides how training data for the ensemble members is chosen or modified
		balancing -- boolean, wether to use balancing as a postprocessing step. defaults to False.
		m -- number of ensemble members. will be automatically incremented by 1 in case method=="residuals".
		k -- size of the generated ensembles used in balancing, must divide m evenly.
		folder -- trained ensemble member sessions will be saved in folder "{folder}/net{i}/", which defaults to "ensemble_{method}/net{i}".
		gpus - tuple of how many parallel processes to assign to which gpu. (3,0,2) will assign 3 processes to gpu0 and 2 processes to gpu2.
		'''

		if method not in ["pairs", "residuals"]:
			raise ValueError("unknown method \"{}\" must be 'pairs' or 'residuals'".format(method))
		if balancing:
			if k < 2:
				raise ValueError("k must be 2 or higher")
			if not (m/k).is_integer():
				raise ValueError("m must be divisible by k without remainder")
			if m/k <= 1:
				raise ValueError("m/k={} generated balancing ensembles, but must be >= 2".format(m/k))
		self.method = method
		self.balancing = balancing
		self.network = network
		self.m = m + (1 if method == "residuals" else 0)
		self.folder = folder if folder else "ensemble_{}".format(method)
		self.k = k

		# multiprocessing management
		mgr = mp.Manager()
		self.gpus = mgr.list([procs for procs in gpus])
		self.lock = mgr.Lock()
		self.pool = mp.Pool(processes=sum(self.gpus), maxtasksperchild=1)

		path = os.path.join('trained_models/', self.folder)
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

		self.ensemble_names = [
			os.path.join(self.folder, "net{}".format(i))
			for i in range(self.m)
		]

	def fit(self, tr, va):
		# see [tibshirani1995 - a comparison of error estimates for neural network 
		# models] for the descriptions of 'bootstrap_pairs' and 
		# 'bootstrap_residual' algorithms
		va = dict(va) # unpack Frame for seemless pickling

		args = []
		if self.method == "pairs":
			# fit all ensemble members, but don't keep them in memory.
			# persistence is expected to be taken care of by the network-shim.
			for i,(I,iname) in enumerate(zip(self._bootstrap(tr), self.ensemble_names)):
				# prepare data for training
				os.makedirs(os.path.join("trained_models", iname), exist_ok=True)
				np.savez(os.path.join("trained_models", iname, "tr.npz"), **tr[I])
				args += [(self.network, iname, va)]

		elif self.method == "residuals":
			# network 0 is the base network to generate residuals
			iname = self.ensemble_names[0]
			os.makedirs(os.path.join("trained_models", iname), exist_ok=True)
			np.savez(os.path.join("trained_models", iname, "tr.npz"), **tr)
			fit_args = (self.network, iname, dict(va))
			predict_args = (self.network, iname, dict(tr)) # predict training set labels
			self.pool.starmap(_assign_gpu, ((self.lock, self.gpus, _fit, fit_args),), 1)
			prY, *_ = self.pool.starmap(_assign_gpu, ((self.lock, self.gpus, _predict, predict_args),), 1)
			print(prY.shape, "prY")
			trR = tr["T"] - prY  # residuals
			print((trR**2).mean()**0.5, 'tr residual RMSE')

			residual_tr = tr.copy()
			for i,(I,iname) in enumerate(zip(self._bootstrap(tr), self.ensemble_names[1:])):
				# distort training set
				residual_tr["T"] = prY + trR[I]
				# prepare data for training
				os.makedirs(os.path.join("trained_models", iname), exist_ok=True)
				np.savez(os.path.join("trained_models", iname, "tr.npz"), **residual_tr)
				args += [(self.network, iname, va)]

		# fit
		rmses = self.pool.starmap(_assign_gpu, ((self.lock, self.gpus, _fit, f_args) for f_args in args), 1)
		print(rmses)

		return self

	def predict(self, data, coverage=None):
		trained = [
			os.path.isfile(os.path.join("trained_models", iname, "session.index"))
			for iname in self.ensemble_names
		]
		if not all(trained):
			names = [name for name,istrained in zip(self.ensemble_names, trained) if not istrained]
			raise RuntimeError("Ensemble wasn't fitted yet, no savefile was found for following networks: \n{}".format(str("\n".join(names))))

		data = dict(data) # unpack Frame for seemless pickling
		args = []
		for iname in self.ensemble_names:
			args += [(self.network, iname, data)]
		# results are returned in proper order
		prYs = self.pool.starmap(_assign_gpu, ((self.lock, self.gpus, _predict, f_args) for f_args in args), 1)

		# exclude the predictions of the first estimator in case of residuals,
		# because that one is used as the base of the residual calculation
		exclude_first = 1 if self.method == "residuals" else 0 
		prY = np.mean(prYs, axis=0)
		# prY = np.median(prYs, axis=0)
		# calc uncertainty
		if self.balancing:
			variance = self._balanced_var(np.array(prYs[exclude_first:]))
		else:
			variance = np.var(prYs[exclude_first:], axis=0, ddof=1)
		std = np.sqrt(variance)

		if coverage is None:
			return prY, prY - std, prY + std
		else:
			# see also [heskes1997:"practical confidence and prediction intervals"]
			df = self.m - (1 if self.method == "residuals" else 0)
			alpha = 1 - coverage
			lower = scipy.stats.t.ppf(alpha/2, df)
			upper = scipy.stats.t.ppf(1-alpha/2, df)
			return prY, prY + lower*std, prY + upper*std

	def _bootstrap(self, data):
		N = len(data)
		I = np.arange(N)
		bootstrapIs = [
			np.random.choice(I, size=N, replace=True)
			for i in range(self.m - (1 if self.method == "residuals" else 0))
		]
		return bootstrapIs

	def _balanced_var(self, ensYs):
		'''
		performs balancing as described in carney1999 or zapranis2014.
		calculates the variance of 1000 bootstrapped groups of {m/k} committes with 
		{k} members. generally results in lower variance.
		params:
			ensYs - numpy array of the predictions made by the ensemble members. it should not include the predictions of the residuals estimator.
		'''
		# k closer to m => lower coverage (lower variance)
		# => k, m, and #resamples become hyperparameters to tune for a certain 
		# target coverage
		# carney used m=200, k=25 and resampled them 1000 times
		# might be a good idea to choose (k) such that the number of ways to choose (k) out of the (m) ensemble members is maximized

		I = np.arange(len(ensYs))
		m = len(I) # size of org. ensemble

		# figure out a good number of splits and groupsize
		# K = np.array([m/k for k in range(3,20)[::-1]])
		# print(K)
		# num_splits = int(K[[k.is_integer() for k in K]][-1])
		num_splits = int(m/self.k)
		num_members = self.k

		splits = np.array([ensYs[s].mean(axis=0) for s in np.split(I, num_splits)])
		groups = (
			splits[np.random.choice(np.arange(num_splits), size=num_splits, replace=True)]
			for _ in range(1000)
		)
		variances = [g.var(axis=0) for g in groups]
		return np.mean(variances, axis=0)
