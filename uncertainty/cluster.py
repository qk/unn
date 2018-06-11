import numpy as np
from utils import Frame, ecdf
from uncertainty.base import UncertaintyEstimator, PipeMixin, MahalanobisUncertaintyMixin
# GaussianDensity
from joblib import Parallel, delayed
from utils import run
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel
from collections import defaultdict
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
# ClusterUncertainty
from collections import defaultdict
# Cmeans
from skfuzzy.cluster import cmeans, cmeans_predict

__all__ = [ 'ClusterUncertainty' ]

class GaussianDensity(BaseEstimator):
	def __init__(self, gamma=None, n_jobs=1):
		self.gamma = gamma
		self.n_jobs = n_jobs

	@staticmethod
	def _dens(X, teX, gamma, i, splits):
		# bosnic2008: raw DENS reliability estimate (without the inversion)
		print("processing {}/{}".format(i, splits), end="\r")
		return rbf_kernel(X, teX, gamma=gamma).mean(axis=0)

	def fit(self, X):
		self.X = X
		return self

	def predict(self, X):
		splits = int(np.ceil(len(X)/1000))
		K = Parallel(self.n_jobs, 'threading')(
			delayed(run)(self._dens, self.X, X[idx], self.gamma, i, splits)
				for i,idx in enumerate(np.array_split(np.arange(len(X)), splits))
		)
		return np.concatenate(K)


class CMeans(BaseEstimator):
	# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
	def __init__(self, n_clusters=None, exp=2, error=0.005, max_iter=1000):
		self.n_clusters = n_clusters
		self.exp = exp
		self.error = error
		self.max_iter = max_iter
		self.centroids = None
	def fit(self, X, Y=None):
		if self.n_clusters is None:
			self.n_clusters = int((len(X)/2)**0.5) # heuristic from pevec2013
		self.labels_ = np.arange(self.n_clusters)
		centroids, u, _, dists, _, _, fpc = cmeans(X.T, self.n_clusters, self.exp, self.error, self.max_iter)
		self.centroids = centroids
		self.fpc = fpc
		# print(n_clusters, "#clusters", self.ref.N, "trN")
		# print(centroids.shape, "centroids") # (n_clusters, n_features)
		# print(u.shape, "u") # (n_clusters, N), membership grades
		# print(dists.shape, "dists") # (n_clusters, N)
		return self
	def predict(self, X):
		u, _, dists, _, _, fpc = cmeans_predict(X.T, self.centroids, self.exp, self.error, self.max_iter)
		return u.T
	def fit_predict(self, X):
		self.fit(X)
		return self.predict(X)

# ns = np.arange(1,30)
# fpcs = np.empty(len(ns))
# for i,n in enumerate(ns):
#	 print("\r{}/{}".format(i,len(ns)), end="")
#	 est = CMeans(n, 2, 0.0001, 5000)
#	 memberships = est.fit_predict(pipe.transform(vaD)) # (N, clusters)
#	 fpcs[i] = est.fpc
# %matplotlib inline
# pp.figure()
# pp.plot(ns, fpcs)
# pp.plot(ns, fpcs, "k.")
# def rbf(X, teX, gamma=None):
#	 if gamma is None:
#		 gamma = teX.shape[-1]
#	 return np.exp(-(sp.spatial.distance.cdist(X, teX)**2)/gamma)


class ClusterUncertainty(UncertaintyEstimator, PipeMixin, MahalanobisUncertaintyMixin):
	"""
	pevec2013: kmeans using heurist k=sqrt(trN/2)
	shrestha2006: cmeans
	you may choose to include the predictions (Y) in the kNN distance calculations
	"""
	# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
	def __init__(self, network_shim, method="kmeans", sets="va", keys="DY", coverage=0.95, n_jobs=1):
		self.network_shim = network_shim
		self.sets = sets
		self.keys = keys
		self.coverage = coverage
		self.n_jobs = n_jobs
		self.method = method
		if sets not in "tr va":
			raise ValueError("sets must specify a reference set, choose from ['tr','va'].")

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

		n_clusters = int((len(self.ref)/2)**0.5) # heuristic from pevec2013
		a = 1.0 - self.coverage
		R = self.ref_results["R"]

		if self.method == 'kmeans' or self.method == 'kmeans-mahalanobis':
			self.cluster_est = KMeans(n_clusters=n_clusters, n_jobs=self.n_jobs)
			memberships = self.cluster_est.fit_predict(self._pipe_transform(self.ref))
			self.ref_results["memberships"] = memberships
			self.clusterbounds = np.empty((n_clusters, 2))
			for c in range(n_clusters):
				cdf, ppf = ecdf(R[memberships == c])
				self.clusterbounds[c,:] = ppf(a/2), ppf(1-a/2)

		elif self.method == "cmeans":
			self.cluster_est = CMeans(n_clusters=n_clusters, error=1e-5, max_iter=10*self.ref.N)
			memberships = self.cluster_est.fit_predict(self._pipe_transform(self.ref)) # (N, clusters)
			asc_residuals = np.argsort(R)
			# self.clusterbounds = np.empty((n_clusters, 2))
			# for c in range(n_clusters):
			#	 cumsum = np.cumsum(memberships.T[c][asc_residuals])
			#	 lb = np.argwhere(cumsum < (a/2)*cumsum[-1])[-1,0]
			#	 ub = np.argwhere(cumsum > (1-a/2)*cumsum[-1])[0,0]
			#	 self.clusterbounds[c,:] = R[asc_residuals][lb], R[asc_residuals][ub]
			cumsum = np.cumsum(memberships[asc_residuals], axis=0).T # (clusters, N)
			self.clusterbounds = np.vstack([
				R[asc_residuals][np.argmax(cumsum >= (a/2)*cumsum[:,-1:], axis=1)],
				R[asc_residuals][np.argmin(cumsum <= (1 - a/2)*cumsum[:,-1:], axis=1) - 1]
			]).T

		elif self.method == 'density':
			self.cluster_est = GaussianDensity(n_jobs=self.n_jobs)
			self.cluster_est.fit(self._pipe_transform(self.ref))
			density = self.cluster_est.predict(self._pipe_transform(self.ref))
			self.max_density = density.max()

		else:
			raise ValueError("Method {} is not a valid option.".format(self.method))

		return self

	def predict_raw(self, D):
		""" returns dictionary of raw results """
		network = self.network_shim()
		prY = np.squeeze(network.predict(D)['Y'])
		pr = Frame(D.items(), Y=prY)

		result = defaultdict(list)
		splits = int(np.ceil(pr.N/5000))
		for i,idx in enumerate(np.array_split(np.arange(pr.N), splits)):
			print("\rworking on split {i} of {n}...".format(i=i+1, n=splits), end="")
			memberships = self.cluster_est.predict(self._pipe_transform(pr[idx]))
			result["memberships"] += [memberships]
		print()
		pr.update([(k, np.concatenate(v)) for k,v in result.items()])

		return pr

	def predict(self, D, coverage=None):
		global M
		""" returns prY, minY, maxY """
		if coverage is not None:
			print("WARNING: ClusterUncertainty requires the desired coverage as an init parameter, otherwise it will be ignored.")
		print("predicting...")
		pr = self.predict_raw(D)
		prY = pr['Y']
		minY = np.empty(len(prY))
		maxY = np.empty(len(prY))

		memberships = pr['memberships']

		if self.method == "kmeans-mahalanobis":
			# can't guarantee a point from every cluster, so will have to use np.unique() here
			X = self._pipe_transform(pr)
			clX = self._pipe_transform(self.ref) # classX
			for c in np.unique(memberships):
				members = (memberships == c)
				cluster = (self.ref_results['memberships'] == c)
				U = self.mahalanobis_uncertainty(X[members], clX[cluster].mean(axis=0, keepdims=True), clX)
				minY[members], maxY[members] = -U, U
			minY, maxY = prY + prY.std()*minY/(2*3*minY.std()), prY + prY.std()*maxY/(2*3*maxY.std())

		elif self.method[:len('kmeans')] == "kmeans":
			# can't guarantee a point from every cluster, so will have to use np.unique() here
			for c in np.unique(memberships):
				members = (memberships == c)
				low, high = self.clusterbounds[c,:]
				minY[members], maxY[members] = prY[members] + low, prY[members] + high

		elif self.method == "cmeans":
			# memberships: (N, clusters)
			# clusterbounds: (clusters, 2)
			bounds = memberships.dot(self.clusterbounds)
			minY = prY + bounds[:,0]
			maxY = prY + bounds[:,1]

		elif self.method == "density":
			density = memberships
			minY = prY - (self.max_density - density)
			maxY = prY + (self.max_density - density)

		return prY, minY, maxY

# est = ClusterUncertainty(CachedShim, coverage=0.95, method='kmeans-mahalanobis', sets="va", keys="DY", n_jobs=n_cores//3)
# est.fit(tr, va)
# ci = te[-20000:]
# prY, minY, maxY = est.predict(ci)
# Y = ci['T']
# %matplotlib inline
# pp.figure(figsize=(10,8))
# pp.subplot(211)
# plot_region(Y, prY, minY, maxY, 1000, 200)
# pp.subplot(212)
# plot_sorted_confidence_error(Y, prY, minY, maxY, center='U')
