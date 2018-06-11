import scipy
import scipy.stats
from uncertainty.base import UncertaintyEstimator

__all__ = [ 'UncertaintyWrapper' ]

class UncertaintyWrapper(UncertaintyEstimator):
	'''
	Simple wrapper, forwards the uncertainty (assumed to be sqrt(variance_measure)) predicted by the network. 
	'''
	def __init__(self, network_shim, ppf=None, name="uncertainty_wrapper"):
		self.name = name
		self.ppf = ppf
		self.network_shim = network_shim
	
	def fit(self, trD, vaD):
		network = self.network_shim(name=self.name)
		network.fit(trD, vaD)
		return self
	
	def predict(self, teD, coverage=None):
		network = self.network_shim(name=self.name)
		pr = network.predict(teD)
		prY = pr['Y']
		std = pr['U'] # sqrt(variance)

		c_lower, c_upper = 1, 1
		if coverage is not None and self.ppf is not None:
			a = 1.0 - coverage
			c_lower = self.ppf(a/2)
			c_upper = self.ppf(1-a/2)
		return prY, prY + std*c_lower, prY + std*c_upper
