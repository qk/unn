import numpy as np
from uncertainty.base import UncertaintyEstimator, PipeMixin
from utils import Frame

# QuantileForestUncertainty
from extra.pyquantregForest.pyquantregForest import QuantileForest

__all__ = [ 'QuantileForestUncertainty' ]


class QuantileForestUncertainty(UncertaintyEstimator, PipeMixin):
	'''
	QuantileForestRegressor as suggested in [pevec2013]. This is a wrapper of an 
	existing implementation pulled from github.
	[pevec2013] did NOT include the networks' prY in the input data for the 
	forest.
	'''
	def __init__(self, network_shim, keys="DY", **keywords):
		self.network_shim = network_shim
		self.keys = keys
		self.est = QuantileForest(**keywords)

	def fit(self, tr, va):
		network = self.network_shim().fit(tr, va)
		self.ref = Frame(va.items(), Y=network.predict(va)['Y'])

		# maybe include the tr set, if prY isn't part of the input
		# self.ref = Frame([(k, np.concatenate([ref[k],tr[k]])) for k in ref.keys()])

		self._pipe_fit(self.ref, self.keys)
		self.est.fit(self._pipe_transform(self.ref), self.ref['T'])
		return self

	def predict(self, D, coverage=0.95):
		prY = self.network_shim().predict(D)['Y']
		pr = Frame(D.items(), Y=prY)
		alpha = 1.0 - coverage

		# opt: SQP or Cobyla
		quantiles = self.est.compute_quantile(self._pipe_transform(pr), np.array([alpha/2, 1.0-alpha/2]), do_optim=True, opt_method='Cobyla')
		# print(quantiles.shape) # (N, 2)

		return prY, quantiles[:,0], quantiles[:,1]

# est = QuantileForestUncertainty(SimpleNetworkShim, keys="D", n_estimators=500, max_leaf_nodes=None, n_jobs=n_jobs)
# est = QuantileForestUncertainty(SimpleNetworkShim, keys="D", n_estimators=500, max_depth=30, n_jobs=n_jobs)
# est.fit(tr, va)
# ci = te[-20000:]
# prY, minY, maxY = est.predict(ci, coverage=0.95)
# Y = ci['T']
# %matplotlib inline
# pp.figure(figsize=(10,8))
# pp.subplot(211)
# plot_region(Y, prY, minY, maxY, 1000, 200)
# pp.subplot(212)
# plot_sorted_confidence_error(Y, prY, minY, maxY, center='U')
