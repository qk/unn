import numpy as np
import os
from sklearn.base import BaseEstimator, RegressorMixin
from utils import exp_lr
from networks.simple import SimpleNetwork
from networks.test_bn_dtnn import BatchNormalizationVarianceDTNN
from networks.density_dtnn import DensityDTNN
from networks.density_simple import SimpleDensityNetwork
from networks.dtnn import DTNN
from networks.mixture_dtnn import DensityMixtureDTNN
from networks.mixture_simple import SimpleMixtureNetwork
from networks.dropout_ho_dtnn import HomoscedasticDTNN
from networks.dropout_he_dtnn import HeteroscedasticDTNN
from abc import ABC, abstractmethod
# cachedshim
from utils import Frame

'''
Neural network shim classes to preset network parameters. A neural network shim 
is a parameterless constructor that returns an object that has fit(tr, va) and 
predict(te) methods. Some uncertainty-estimators will NOT persist shims between 
fit() and predict() calls due to resource constraints. Thus trained networks 
should be saved to disk in fit() and loaded in predict() for these uncertainty 
estimators.

Any uncertainty estimator works with any shim class and most networks should be 
wrappable in shim classes.

Module dictionary 'params' can be used to persist network parameters between 
method calls.
'''

# shims
#  + flexible, most classes can be wrapped into a shim
#  + fewest LOC (m+n): write uncertainty_estimators once (m), shims once (n), combine them during uncertainty-estimator initialization
#  + new network => 1 new shim
#  + uncertainty estimators can reasonably expect the shims to be parameterless, simplifying their implementation considerably
#  - dirty
# ensemble_uncertainty(network)
#  + straightforward class model
#  - any uncertainty estimator should work with any network. implementing it this way you'd have to copy/paste the ensemble-estimator #network times (m*n classes)
# - new network => m new classes
# ensemble_uncertainty(network, ensemble_mixin) 
#  + clean python inheritence model
#  - have to define a new class per uncertainty-network-combination (m*n-ish short classes)
#  - new network => m new shim-like classes

# global defaults
params = dict()

def _assert_exist(keys, announce=False):
	for p in keys.split():
		if p not in params.keys():
			raise RuntimeError("'{}' must be defined in global dict 'params' of the shims module.".format(p))
	if announce:
		if 'tr_mean_std' in keys:
			print(params['tr_mean_std']["E"].mean(), "mean of E tr_mean_std")
		if 'point' in keys:
			print(params['point']["E"].shape, "shape of E point")
		if 'mu_max' in keys:
			print(params['mu_max'], "mu_max")

class SaveLoadShim(ABC, BaseEstimator, RegressorMixin):
	'''
	Convenience class to persist models on disk between fit() and predict() calls 
	and to automatically avoid retraining already trained models.
	'''
	def __init__(self, name="saveloadshim", save=True):
		self.name = name
		self.save = save
		self.net = None # will be set if saving is disabled
		self.historyfile = None
	@abstractmethod
	def _create(self):
		''' creates a network instance and returns it '''
		pass
	@abstractmethod
	def _train(self, net, tr, va):
		''' fits the instantiated network "net" and returns it '''
		pass
	@abstractmethod
	def _unwrap(self, result):
		'''
		unifies labels of the result dictionary.
		target values should be 'Y'.
		uncertainty values should be 'U'.
		'''
		pass
	def fit(self, tr, va):
		if self.is_trained(tr, va):
			print("{} is already trained".format(self.name))
			return self
		net = self._create()
		# make sure we train on "T"
		# ensemble with method=residuals modifies the labels
		tr = tr.copy()
		tr["E"] = tr["T"]
		net = self._train(net, tr, va)
		self.historyfile = net.historyfile
		if self.save:
			net.save()
			net.close()
		else:
			self.net = net
		self.save_training(tr, va)
		return self
	def predict(self, d):
		if self.save:
			net = self._create()
			net.load()
			result = net.predict(d)
			net.close()
		else:
			result = self.net.predict(d)
		return self._unwrap(result)
	def save_training(self, tr, va):
		path = os.path.join("trained_models", self.name, "last_training_labels.npz")
		# save targets, try to be permutation invariant
		np.savez(
			path,
			name=self.name,
			trT=np.sort(tr["E"]),
			vaT=np.sort(va["E"])
		)
	def is_trained(self, tr, va):
		# checks if this network is already trained
		# pretty crude, but good enough if you're aware of how it's checked
		path = os.path.join("trained_models", self.name, "last_training_labels.npz")
		if not os.path.exists(path):
			return False
		last_data = np.load(path)
		if last_data["name"] != self.name:
			print("names are different ({} != {})".format(last_data["name"], self.name))
			return False
		if not np.all(last_data["trT"] == np.sort(tr["E"])):
			print("training set labels are different, assuming network not trained")
			return False
		if not np.all(last_data["vaT"] == np.sort(va["E"])):
			print("validation set labels are different, assuming network not trained")
			return False
		return True

class DTNNShim(SaveLoadShim):
	def __init__(self, name="dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return DTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			10 # epoch
		)
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"]}

class DensityShim(SaveLoadShim):
	def __init__(self, name="density_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return DensityDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.density_dtnn import FullDensityDTNN
class FullDensityShim(SaveLoadShim):
	def __init__(self, name="density_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return FullDensityDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.density_dtnn import SteppedDensityDTNN
class SteppedDensityShim(SaveLoadShim):
	def __init__(self, name="density_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return SteppedDensityDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.density_dtnn import HalfDensityDTNN
class HalfDensityShim(SaveLoadShim):
	def __init__(self, name="half_density_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return HalfDensityDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.mixture_dtnn import HalfDensityMixtureDTNN
class HalfMixtureShim(SaveLoadShim):
	def __init__(self, name="half_density_mixture_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return HalfDensityMixtureDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.mixture_dtnn import FullDensityMixtureDTNN
class FullMixtureShim(SaveLoadShim):
	def __init__(self, name="full_density_mixture_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return FullDensityMixtureDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4200, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.mixture_dtnn import SteppedDensityMixtureDTNN
class SteppedMixtureShim(SaveLoadShim):
	def __init__(self, name="stepped_density_mixture_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return SteppedDensityMixtureDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

from networks.mixture_dtnn import ReDensityMixtureDTNN
class ReMixtureShim(SaveLoadShim):
	def __init__(self, name="re_density_mixture_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return ReDensityMixtureDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class MixtureShim(SaveLoadShim):
	def __init__(self, name="mixture_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point mu_max", announce=True)
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		return DensityMixtureDTNN(*args, **keywords)
	def _train(self, net, tr, va):
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 4000, lr=lr, silent=True)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HeteroscedasticShim90(SaveLoadShim):
	def __init__(self, name="heteroscedastic80_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std n_tr_points point mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.9
		net = HeteroscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HeteroscedasticShim80(SaveLoadShim):
	def __init__(self, name="heteroscedastic80_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std n_tr_points point mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.8
		net = HeteroscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HeteroscedasticShim70(SaveLoadShim):
	def __init__(self, name="heteroscedastic90_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std n_tr_points point mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.7
		net = HeteroscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HeteroscedasticShim60(SaveLoadShim):
	def __init__(self, name="heteroscedastic90_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std n_tr_points point mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.6
		net = HeteroscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HomoscedasticShim60(SaveLoadShim):
	def __init__(self, name="homoscedastic70_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point n_tr_points mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.6
		net = HomoscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HomoscedasticShim70(SaveLoadShim):
	def __init__(self, name="homoscedastic70_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point n_tr_points mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.7
		net = HomoscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HomoscedasticShim80(SaveLoadShim):
	def __init__(self, name="homoscedastic80_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point n_tr_points mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.8
		net = HomoscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

class HomoscedasticShim90(SaveLoadShim):
	def __init__(self, name="homoscedastic90_dtnn_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point n_tr_points mu_max", announce=True)
		args = params['tr_mean_std'], params['point'], params['n_tr_points']
		keywords = dict(name=self.name, mu_max=params['mu_max'])
		keywords['keep_prob'] = 0.9
		net = HomoscedasticDTNN(*args, **keywords)
		return net
	def _train(self, net, tr, va):
		restart = (
			"rmse", # loss op name
			0.5*(((va['T'] - va['T'].mean())**2).mean()**0.5), # threshold
			30 # epoch
		)
		stop = {'rmse':0.06}
		lr = exp_lr(1e-3, 1e-4, 3000, floor=1e-4)
		net.fit(tr, va, 25, 12000, lr=lr, silent=True, restart_check=restart, stop_at=stop)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"], "U":result["U"]}

### unofficial testing starts here #############################################

# class Fixed(SaveLoadShim):
	# # lr fixed, 3k epochs a 25 batches -> use with adam
	# def __init__(self, name="dtnn_fixed_lr", save=True):
		# super().__init__(name=name, save=save)
	# def _create(self):
		# _assert_exist("tr_mean_std point")
		# args = params['tr_mean_std'], params['point']
		# keywords = dict(name=self.name)
		# return DTNN(*args, **keywords)
	# def _train(self, net, tr, va):
		# net.fit(tr, va, 25, 3000, lr=1e-3, silent=True)
		# return net
	# def _unwrap(self, result):
		# return {"Y":result["E"]}

# class Decayed(SaveLoadShim):
	# # lr exp decayed, 3k epochs a 25 batches -> use with momentum
	# def __init__(self, name="dtnn_decayed_momentum", save=True):
		# super().__init__(name=name, save=save)
	# def _create(self):
		# _assert_exist("tr_mean_std point")
		# args = params['tr_mean_std'], params['point']
		# keywords = dict(name=self.name)
		# return DTNN(*args, **keywords)
	# def _train(self, net, tr, va):
		# lr = exp_lr(1e-3, 1e-4, 2000, floor=1e-5)
		# net.fit(tr, va, 25, 3000, lr=lr, silent=True)
		# return net
	# def _unwrap(self, result):
		# return {"Y":result["E"]}

class SimpleDropoutShim(BaseEstimator, RegressorMixin):
	def __init__(self, name=None, save=True):
		self.name = name or "simple_dropout_shim"
		self.save = save
		self.net = None # will be set if saving is disabled
	def fit(self, tr, va):
		net = SimpleDropout(params['tr_mean_std'], params['point'], name=self.name)
		net.fit(tr, va, 25, 120, silent=False, lr=1e-3)
		if self.save:
			net.save()
			net.close()
		else:
			self.net = net
		return self
	def predict(self, d):
		if self.save:
			net = SimpleDropout(params['tr_mean_std'], params['point'], name=self.name)
			net.load()
			result = net.predict(d)
			net.close()
		else:
			result = self.net.predict(d)
		return {"Y":result["E"], "U":result["U"]}

class SimpleShim(SaveLoadShim):
	def __init__(self, name="simple_preset", save=True):
		super().__init__(name=name, save=save)
	def _create(self):
		_assert_exist("tr_mean_std point")
		args = params['tr_mean_std'], params['point']
		keywords = dict(name=self.name)
		return SimpleNetwork(*args, **keywords)
	def _train(self, net, tr, va):
		lr = exp_lr(1e-3, 1e-5, 1000, floor=5e-5)
		net.fit(tr, va, 200, 30, lr=lr, silent=True)
		return net
	def _unwrap(self, result):
		return {"Y":result["E"]}

# class SimpleShim(BaseEstimator, RegressorMixin):
	# def fit(self, tr, va):
		# self.network = SimpleNetwork(preproc.scaling_params, tr["D"].shape[1], gpu=0)
		# self.network.fit(tr, va, 200, 3000, silent=True, savename="uncertainty_dummy", lr=1e-3)
		# self.network.close()
		# return self
	# def predict(self, D):
		# self.network = SimpleNetwork(preproc.scaling_params, D["D"].shape[1], gpu=0)
		# result = self.network.predict_on_scale(D, savename="uncertainty_dummy")
		# self.network.close()
		# return {"Y":result["E"]}

class CachedShim(BaseEstimator, RegressorMixin):
	'''
	requires:
	  * data to be Frames, that have an 'index' array, indicating the index of each point
	  * testdata to be available at './tmp/cachedshim.npz'
	predict returns cachedata entries where 'index' values match those of the testdata
	'''
	def __init__(self, cache='tmp/cachedshim.npz'):
		self.cachefile = cache
	def fit(self, tr, va):
		return self
	def predict(self, data):
		cache = Frame(**np.load(self.cachefile))
		cacheindex = cache['index'] # should be sorted
		I = np.argsort(cacheindex)
		return cache[I[np.searchsorted(cacheindex[I], data['index'])]]
