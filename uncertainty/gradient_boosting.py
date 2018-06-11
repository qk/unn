import numpy as np
from utils import Frame
from uncertainty.base import UncertaintyEstimator, PipeMixin
from sklearn.ensemble import GradientBoostingRegressor

__all__ = [ 'GradientBoostingUncertainty' ]

class GradientBoostingUncertainty(UncertaintyEstimator, PipeMixin):
	def __init__(self, network_shim, coverage=0.95, keys="DY", **keywords):
		self.network_shim = network_shim
		self.keys = keys
		self.est = GradientBoostingRegressor(**keywords)
		self.est_min = GradientBoostingRegressor(**keywords)
		self.est_max = GradientBoostingRegressor(**keywords)
		self.est.set_params(loss='ls')
		alpha = 1.0 - coverage
		self.est_max.set_params(loss='quantile', alpha = 1 - alpha/2) # 1/2 + c/2
		self.est_min.set_params(loss='quantile', alpha = alpha/2) # 1/2 - c/2
		
	def fit(self, tr, va):
		network = self.network_shim().fit(tr, va)
		self.ref = Frame(va.items(), Y=network.predict(va)['Y'])
		
		self._pipe_fit(self.ref, self.keys)
		print("fitting normal", end="\r")
		self.est.fit(self._pipe_transform(self.ref), self.ref['T'])
		print("fitting lower ", end="\r")
		self.est_min.fit(self._pipe_transform(self.ref), self.ref['T'])
		print("fitting upper", end="\r")
		self.est_max.fit(self._pipe_transform(self.ref), self.ref['T'])
		print("fitting done")
		return self
	
	def predict(self, D, coverage=None):
		if coverage is not None:
			print("WARNING: GradientBoostingUncertainty requires the desired coverage as an init parameter, otherwise it will be ignored.")

		prY = self.network_shim().predict(D)['Y']
		pr = Frame(D.items(), Y=prY)
		
		prY = self.est.predict(self._pipe_transform(pr))
		minY = self.est_min.predict(self._pipe_transform(pr))
		maxY = self.est_max.predict(self._pipe_transform(pr))
		
		return prY, minY, maxY

# est = GradientBoostingUncertainty(SimpleNetworkShim, n_estimators=100, max_depth=300, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
# est.fit(tr, va)
# te = ciD[-10000:]
# prY, minY, maxY = est.predict(te)
# Y = te['T']
# %matplotlib inline
# pp.figure(figsize=(10,8))
# pp.subplot(211)
# plot_region(Y, prY, minY, maxY, 1000, 200)
# pp.subplot(212)
# plot_sorted_confidence_error(Y, prY, minY, maxY, center='U')
