import numpy as np
from uncertainty.base import UncertaintyEstimator

__all__ = [ 'DummyUncertainty' ]

class DummyUncertainty(UncertaintyEstimator):
	def __init__(self, network_shim, method="const"):
		self.method = method
		self.network_shim = network_shim
		if method not in "const f' 1/f' f'' 1/f''".split(" "):
			raise ValueError("No such dummy: {}".format(method))
	
	def fit(self, tr, va):
		network = self.network_shim()
		network.fit(tr, va)
		if self.method == "const":
			prY = np.squeeze(network.predict(va)["Y"])
			Y = np.squeeze(va["T"])
			self.varY = (Y - prY).var()
		return self
	
	def predict(self, data, coverage=None):
		if coverage is not None:
			print("WARNING: DummyUncertainty ignores the coverage parameter.")
		network = self.network_shim()
		prY = network.predict(data)["Y"]
		if self.method == "const":
			prU = np.repeat(self.varY, len(prY), axis=0)
		elif self.method[-2:] == "f'":
			# assumes chronological ordering
			prU = np.pad(prY[2:] - prY[:-2], (1,1), "edge")
			prU /= 1.5*prU.std()
			prU = np.abs(prU)
			if self.method[:2] == "1/": prU = 1/(0.5 + prU)
		elif self.method[-3:] == "f''":
			# assumes chronological ordering
			prU = np.pad(prY[2:] - prY[:-2], (1,1), "edge")
			prU /= 1.5*prU.std()
			prU = np.pad(prU[2:] - prU[:-2], (1,1), "edge")
			prU /= 1.5*prU.std()
			prU = np.abs(prU)
			if self.method[:2] == "1/": prU = 1/(0.5 + prU)
		return prY, prY - prU, prY + prU

# %matplotlib inline
# methods = "const f' 1/f' f'' 1/f''".split(" ")
# estimators = [DummyUncertaintyEstimator(SimpleNetworkShim, method) for method in methods]
# estimators[0].fit(tr, va)
# P = [(est.method, dict_predict(est, ciD)) for est in estimators]
# for method, pr in P:
#	 pp.figure()
#	 pp.title(method)
#	 plot_sorted_confidence_error(pr['T'], pr['Y'], pr['minY'], pr['maxY'], zoom=False)
#	 pp.tight_layout()
#	 pp.show()
