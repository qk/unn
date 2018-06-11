import numpy as np
import scipy as sp
import scipy.stats
import time

from sklearn.base import BaseEstimator
from utils import ecdf
from plots import plot_coverage_alignment
# PipeMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
# abstract classes
from abc import ABC, abstractmethod

# __all__ is a list of public objects of that module, as interpreted by import *. It overrides the default of hiding everything that begins with an underscore.
__all__ = [
	'UncertaintyEstimator',
	'MahalanobisUncertaintyMixin',
	'PipeMixin',
]


class UncertaintyEstimator(ABC):
	# base class to document the interface common to all uncertainty estimators
	@abstractmethod
	def fit(self, trD, vaD): pass # returns self
	@abstractmethod
	def predict(self, teD, coverage=None): pass # returns prY, minY, maxY (e.i. predicted values and corresponding uncertainty estimates as lower and upper bounds, which should capture 95% of signed errors if possible)
	def dict_predict(self, D, silent=False): # convenience method
		prY, minY, maxY = self.predict(D)
		pr = {}
		pr['T'] = np.squeeze(D["Y"]) # true value (prediction target)
		pr['Y'] = np.squeeze(prY) # predicted value
		pr['minY'] = np.squeeze(minY)
		pr['maxY'] = np.squeeze(maxY)
		pr['U'] = np.abs(pr['maxY'] - pr['minY'])
		if not silent: print("T,Y,U shapes:\n",pr['T'].shape, pr['Y'].shape, pr['U'].shape)
		error = ((pr['T'] - pr['Y'])**2).mean()**0.5
		if not silent: print("squared error\n", error)
		return pr
	def calibrate(self, Y, prY, prU, ppf=sp.stats.norm.ppf):
		'''
		generates an empirical point percentile function to use for uncertainties' 
		coverage of residuals (see self.eppf) and calculates a global misalignment 
		factor between the empirical ppf and target ppf.
		returns -- global misalignment factor between the the empirical ppf and target ppf
		'''
		R = np.abs(Y-prY) # abs. residuals
		cdf, self._ppf = ecdf(R/prU)
		_,X = ecdf(R/prU, interpolate=False)
		# factor = (self._ppf(X)/ppf((1+X)/2)).mean()
		p = np.arange(1,1+len(X))/len(X)
		factor = (np.sort(R/prU)/ppf((1+p)/2)).mean()
		self._mean_scaling_factor = factor

		return factor
	def eppf(self, coverage):
		'''
		empirical point percentile function of abs.residuals divided by uncertainties
		returns a factor to scale the uncertainty by to get the desired coverage of residuals
		( |Y-prY| <= eppf(coverage)*U ).mean() ~= coverage
		'''
		if not hasattr(self, '_ppf'):
			raise RuntimeError("estimator must be calibrated first")
		return self._ppf(coverage)
	def scale(self, prY, prU, coverage=0.95, ppf=None):
		''' scales uncertainty bounds to fit target coverage of residuals '''
		if not hasattr(self, '_ppf'):
			raise RuntimeError("estimator must be calibrated first")
		if ppf is not None:
			prU = self._mean_scaling_factor*ppf(coverage)*prU
		else:
			prU = self._ppf(coverage) * prU
		return prY, prY - prU, prY + prU
	def predict_calibrated(self, va, data, coverage=0.95):
		if self._ppf is None:
			# don't care about coverage in this step, will calibrate anyhow
			prY, minY, maxY = self.predict(va, coverage=0.9)
			prU = (maxY - minY)/2 # most times sth. like std
			factor = self.calibrate(va["T"], prY, prU)
			# predict
			prY, minY, maxY = self.predict(data)
			prU = (maxY - minY)/2 # most times sth. like std
		return self.scale(prY, prU, coverage=coverage)


class MahalanobisUncertaintyMixin:
	'''
	toplak2014:
		uncertainty estimate1: distance(X, knnX).sum()
		uncertainty estimate2: distance(X, trX.mean(axis=0, keepdims=True))
	'''
	@staticmethod
	def _mahalanobis_uncertainty_job(u, V, VI, sum_axis=1):
		return sp.spatial.distance.cdist(u, V, VI=VI, metric='mahalanobis').sum(axis=sum_axis)
	
	def mahalanobis_uncertainty(self, U, V, trX=None, reg=1e-8, n_jobs=1):
		U = U.reshape(len(U), -1)
		V = V.reshape(len(V), -1)
		if not hasattr(self, '_mahalanobis_params'): # init
			if trX is None or np.all(trX == 0):
				raise ValueError('trX is None or all zeroes, cannot compute covariance matrix (maybe caching the covariance matrix inverse failed)')
			trX = trX.reshape(len(trX), -1)
			S = np.cov(trX.T) # V.shape = (#points, ...)
			VI = np.linalg.inv(S + np.eye(len(S))*reg) 
			self._mahalanobis_params = dict(n_jobs=n_jobs, VI=VI)

		VI = self._mahalanobis_params["VI"]
		n_jobs = self._mahalanobis_params['n_jobs']
		
		# try to split the larger array
		switched = len(V) > len(U)
		sum_axis = 1
		if switched:
			U, V = V, U
			sum_axis = 0

		n = 100 if len(V) >= n_jobs*100 else 1000 # allow more points per u if V is small
		splits = max(1, len(U)//n)
		# print(n_jobs, "jobs,", splits, 'splits,', 'switched' if switched else 'same')
		if splits == 1: # not when n_jobs == 1, because splitting may be done to conserves memory
			return self._mahalanobis_uncertainty_job(U, V, VI, sum_axis=sum_axis)
		else:
			D = Parallel(n_jobs, 'threading', verbose=5 if len(V) > 1 else 0)(
				delayed(run)(self._mahalanobis_uncertainty_job, u, V, VI, sum_axis=sum_axis)
				for u in np.array_split(U, splits)
			)
			if switched:
				return np.sum(D, axis=0)
			else:
				return np.concatenate(D) # (len(U), ...)

# mdist = MahalanobisUncertaintyMixin()
# D0 = mdist.mahalanobis_uncertainty(te['D'][-1000:], va['D'][:2000], tr['D'], n_jobs=n_jobs)
# print(D0.shape)

# X = te['D'][-3:].reshape(3, -1)
# refX = va['D'][:4000]
# refX = refX.reshape(len(refX), -1)
# print(X.shape, refX.shape)
# d = mdist.mahalanobis_uncertainty(X, refX)
# print(d.shape)

# D1 = mdist.mahalanobis_uncertainty(te['D'][-4000:], va['D'][:2000], tr['D'], n_jobs=n_jobs)
# print(D1.shape)
# print(np.allclose(D0, D1[-1000:]), np.all(D0 == D1[-1000:]))
# print(np.max(np.abs(D0 - D1[-1000:])))
# pp.subplot(1,2,1)
# pp.hist(D0.ravel())
# pp.subplot(1,2,2)
# pp.hist(D1[-1000:].ravel())


def _interatomic_distances(X):
	# print(X.shape, "X")
	dists = ((X[:,None,:,:] - X[:,:,None,:])**2).sum(axis=-1)**0.5
	return dists.reshape(len(dists), -1)

class PipeMixin:
	def _pipe_fit(self, D, keys):
		# print(D['R'].shape, "dr")
		# TODO: maybe use gower's distance instead
		select = lambda key: lambda D: D[key]
		flatten = lambda X: X.reshape(len(X), -1).astype(float)
		atomic_dists_pipe = ('atomic_dists', Pipeline([
				("select", FunctionTransformer(select('R'), validate=False)),
				('dists', FunctionTransformer(_interatomic_distances, validate=False)),
				("flatten", FunctionTransformer(flatten, validate=False)),
				("scale", StandardScaler()),
				('filter', VarianceThreshold(0.0))
			])
		)
		# select = lambda key: lambda D: D[key].reshape(D.N, -1).astype(float)
		key_pipes = [
			(key, Pipeline([
				("select", FunctionTransformer(select(key), validate=False)),
				("flatten", FunctionTransformer(flatten, validate=False)),
				("scale", StandardScaler()),
				('filter', VarianceThreshold(0.0))
			])) for key in keys
		]
		self._pipe = Pipeline([
			("union", FeatureUnion([atomic_dists_pipe] + key_pipes)),
		])
		self._pipe.fit(D)
		return self
	def _pipe_transform(self, D):
		return self._pipe.transform(D)
