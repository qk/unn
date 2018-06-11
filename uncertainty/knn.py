import numpy as np
from utils import Frame
from collections import defaultdict
from uncertainty.base import UncertaintyEstimator, PipeMixin, MahalanobisUncertaintyMixin
## Clustering
from sklearn.neighbors import NearestNeighbors
## mahalanobis distance
from joblib import Parallel, delayed
from utils import run
from scipy.stats import norm
from scores import ncs

__all__ = [ 'KNNUncertainty' ]

class KNNUncertainty(UncertaintyEstimator, PipeMixin, MahalanobisUncertaintyMixin):
	"""
	pevec2013 uses unweighted residuals 
	pevec2013 is a simplified version of briesemeister2012 CONFINE/CONFIVE
	bosnic2008 gaussian kernel based estimate
	you may choose to include the predictions (keys='Y') in the kNN distance calculations
	"""
	def __init__(self, network_shim, method="label_var", sets="va", keys="", neighbors=20, n_jobs=1, silent=False):
		"""
		params
		------
		neighbors - float, int
			percent of reference points if float
			number of neighbors if int
		"""
		self.network_shim = network_shim
		self.sets = sets
		self.keys = keys
		if not isinstance(neighbors, (int, np.integer)):
			raise ValueError("neighbors parameter must be an integer")
		if neighbors < 2:
			raise ValueError("neighbors must be 2+")
		self.neighbors = neighbors
		self.silent = silent
		self.n_jobs = n_jobs
		self.method = method.lower()
		if sets not in "tr va":
			raise ValueError("sets must specify a reference set, choose from ['tr','va'].")
		legal_methods = [
			"absdev",
			'avgdist',
			'biased_std',
			'dev',
			'label_var',
			'label_std',
			'mae',
			'mahalanobis',
			'mse',
			'rmse',
		]
		if self.method not in legal_methods:
			raise ValueError(
				"no such method {}, choose from {}".format(method, str(legal_methods))
			)

	def fit(self, trD, vaD):
		network = self.network_shim().fit(trD, vaD)
		
		data = []
		if "tr" in self.sets:
			data += [(trD, np.squeeze(network.predict(trD)['Y']))]
		if "va" in self.sets:
			data += [(vaD, np.squeeze(network.predict(vaD)['Y']))]
		
		self.ref = Frame([(k, np.concatenate([D[k] for D,prY in data])) for k in trD.keys()])
		self.ref['Y'] = np.squeeze(np.concatenate([prY for D,prY in data]))
		self.ref_results = Frame(R=self.ref['T']-self.ref['Y']) # residuals
		
		self._pipe_fit(self.ref, self.keys)
		self.knn = NearestNeighbors(n_neighbors=self.neighbors, n_jobs=self.n_jobs)
		self.knn.fit(self._pipe_transform(self.ref))
		
		return self
	
	def predict_raw(self, D, skip_closest=0):
		""" returns dictionary of raw results """
		network = self.network_shim()
		prY = np.squeeze(network.predict(D)['Y'])
		pr = Frame(D.items(), Y=prY)

		if not hasattr(self, "_pipe"):
			raise RuntimeError("Estimator is not fitted yet. The neural network shim should decide if training is necessary or not.")
		X = self._pipe_transform(pr)
		refX = self._pipe_transform(self.ref)

		if self.method == 'mahalanobis': # takes a long time
			if not self.silent: print("caching inverse covariance matrix ...")
			self.mahalanobis_uncertainty(X[:2], refX[:2], refX, n_jobs=self.n_jobs)

		k = self.neighbors
		knn = self.knn
		knn.set_params(n_neighbors=k)

		result = defaultdict(list)
		splits = int(np.ceil(len(pr)/5000))
		for i,idx in enumerate(np.array_split(np.arange(len(pr)), splits)):
			if not self.silent: print("working on split {i} of {n}...".format(i=i+1, n=splits), end="\r")
			distances, neighbors = (A[:,skip_closest:] for A in knn.kneighbors(X[idx]))
			knnR = self.ref_results['R'][neighbors] # signed residuals in neighborhood
			knnY = self.ref['Y'][neighbors] # predictions in neighborhood
			knnT = self.ref['T'][neighbors] # labels (targets) in neighborhood
			# print(knnY.shape, "knnY shape") # (n_points_in_split, n_neighbors)
			result['avgDist'] += [distances.mean(axis=1)]
			result['meanR'] += [knnR.mean(axis=1)] # for stdR
			result['meanY'] += [knnY.mean(axis=1)]
			result['meanT'] += [knnT.mean(axis=1)]
			result['stdR'] += [(((knnR - result['meanR'][-1][:,None])**2).sum(axis=1)/(k - 1))**0.5]
			result['absR'] += [np.abs(knnR).mean(axis=1)]
			result['varT'] += [np.var(knnT, axis=1, ddof=1)] # cs_knnV
			result['stdT'] += [np.var(knnT, axis=1, ddof=1)**0.5] # cs_knnV**0.5
			result['sqR'] += [(knnR**2).mean(axis=1)] # cs_knnE
			if self.method == 'mahalanobis': # takes a long time, exceeds memory limits if not split like this
				VI = self._mahalanobis_params['VI'] # precomputed above
				dists = Parallel(self.n_jobs, 'threading', verbose=0)(
					delayed(run)(
						self._mahalanobis_uncertainty_job,
						X[idx[i]][None,:],
						refX[neighbors[i]],
						VI=VI
					)
					for i in range(len(idx))
				)
				result['mn-dist'] += [np.concatenate(dists)]

		if not self.silent: print()
		pr.update([(k, np.concatenate(v)) for k,v in result.items()])
		return pr

	def predict_uncertainty(self, D, skip_closest=0):
		pr = self.predict_raw(D, skip_closest=skip_closest)
		
		method = self.method
		center = pr['Y']
		if method == 'avgdist': # briesemeister2012
			U = pr['avgDist']
		elif method == 'label_var': # briesemeister2012 CONFIVE
			U = pr['varT']
		elif method == 'label_std': # briesemeister2012 CONFIVE**.5
			U = pr['stdT']
		elif method == 'mse': # briesemeister2012 CONFINE
			U = pr['sqR']
		elif method == 'biased_std': # pevec2013
			# not a mistake, these intervals are not centered on prY
			center = pr['meanR']
			U = pr['stdR']
		elif method == 'dev': # bosnic2008: CNK (deviation from avg. label in the neighborhood)
			# makes the KNN predictor appear the most confident (falsely)
			U = pr['meanT'] - pr['Y']
		elif method == "absdev": # bosnic2008: CNK (deviation from avg. label in the neighborhood)
			# makes the KNN predictor appear the most confident (falsely)
			U = np.abs(pr['meanT'] - pr['Y'])
		elif method == 'mae':
			U = pr['absR']
		elif method == 'rmse':
			U = pr['sqR']**0.5
		elif method == 'mahalanobis': # toplak2014
			U = pr['mn-dist']

		return pr['Y'], center, U

	def predict(self, D, coverage=None):
		""" returns prY, minY, maxY """
		prY, center, U = self.predict_uncertainty(D)

		alpha = 1 - coverage if coverage is not None else 0.6827 # 1 std radius

		if self.method == 'biased_variance':
			return prY, center + norm.ppf(alpha/2)*U, center + norm.ppf(1-alpha/2)*U

		if coverage is None:
			print('using raw knn uncertainty estimates')
			return prY, center - U, center + U
		else: # use method as in briesemeister2012 to create empirical quantiles
			print("postprocessing knn uncertainty to generate quantiles")
			ref_prY, _, ref_U = self.predict_uncertainty(self.ref, skip_closest=1)
			Q = self.quantiles(U, ref_U, self.ref['T'] - ref_prY, alpha=1-coverage) # (2,N)
			return prY, prY + Q[0], prY + Q[1]

	def quantiles(self, U, ref_U, ref_R, alpha=0.05):
		"""
		computes quantiles for uncertainty scores U as described in briesemeister2012
		params
		------
		U - test set uncertainty scores
		ref_U - reference set uncertainty scores (training and/or validation set)
		ref_R - residuals (Y - prY) of the reference set
		alpha - tolerance percent error
		"""
		# not sure what normalizing them is supposed to achieve
		normed_scores = ncs(U, ref_U) # higher is better
		reference_scores = ncs(ref_U, ref_U)
		asc = np.argsort(reference_scores)
		positions = np.searchsorted(reference_scores[asc], normed_scores)

		n = 50 # same as briesemeister2012
		l = len(reference_scores)
		return np.vstack([
			np.percentile(
				ref_R[asc[max(pos-n,0):min(pos+n,l)]],
				[100*alpha/2, 100*(1-alpha/2)]
			)
			for pos in positions
		]).T


# est = KNNUncertainty(CachedShim, method='avgDist', sets="va", p=0.2, keys="DY", calibrate=True, n_jobs=n_jobs)
# est.fit(tr, va)
# ci = te[-20000:]
# est.method = 'avgDist'
# # prY, minY, maxY = est.predict(ci)
# pr = est.predict_raw(ci)
# R = est.ref_results['R']
# Y = ci['T']
# %matplotlib inline
# pp.figure(figsize=(10,8))
# pp.subplot(211)
# plot_region(Y, prY, minY, maxY, 1000, 200)
# pp.subplot(212)
# plot_sorted_confidence_error(Y, prY, minY, maxY, center='U')
