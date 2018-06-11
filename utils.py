import numpy as np
from matplotlib import pyplot as pp
import scipy as sp
import scipy.stats
from scipy import spatial
import tensorflow as tf
from datetime import datetime
from time import time, sleep
import re

# class tf_reg:
	# l1 = tf.contrib.layers.l1_regularizer
	# l2 = tf.contrib.layers.l2_regularizer

# class tf_configs:
	# useCPU = tf.ConfigProto(device_count = {'GPU': 0})
	# saveMemory = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True))

# class tf_inits:
	# normal = tf.contrib.layers.variance_scaling_initializer()
	# uniform = tf.contrib.layers.variance_scaling_initializer(uniform=True)
	# zeros = tf.constant_initializer(0)
	# ones = tf.constant_initializer(1)

def run(f, *args, **keywords):
	'''
	helper for
		Parallel(n_jobs, 'threading')(
			delayed(run)(f, *args, **keywords) for ...)
	accepts any function reference as f, doesn't try any pickling
	'''
	return f(*args, **keywords)

class Frame(dict):
	"""
	wraps a dictionary of numpy arrays, makes it sliceable.
	i find it more straightforward to use than pandas.
	
	F = Frame(I=np.arange(10), V=np.arange(10))
	print(F[np.array([0,2,4])], "entries 0,2,4")
	print(F["I"], "array for key 'I'")
	print(F.keys(), "keys")
	F["V"] = np.arange(10)/10
	print(F, "changed array for key V")
	F2 = Frame(I=np.array([-1, -1, -1]))
	print(F2, "F2")
	F[np.array([1,3,5])] = F2
	print(F, "F with 1,3,5 set to -1 in array 'I'")
	"""
	def __init__(self, *args, **keywords):
		super(self.__class__, self).__init__(*args, **keywords)
		# assert all arrays have same length
		lengths = [len(v) for v in self.values()]
		if lengths and not lengths.count(lengths[0]) == len(lengths):
			lengths = ", ".join(("'{}':{}".format(k, len(v))for k,v in self.items()))
			raise ValueError("Axis 0 must be of equal size among arrays: {}.".format(lengths))
		self._N = lengths[0] if lengths else 0
	@property
	def N(self):
		if not hasattr(self, "_N"): # happens if un-pickled
			# don't support pickling right now. i only need it for multiprocessing 
			# and you have to phrase your imports just right for pickle to work with 
			# multiprocessing. so right now the most stable strategy is to unpack the 
			# frame into a dict and repack the dict into a frame in the forked 
			# process. it's just dict(myframe) or Frame(mydict) respectively.
			# it's easy enough to restore _N like this
			# https://stackoverflow.com/questions/11801284/how-to-overwrite-the-dump-load-methods-in-the-pickle-class-customizing-picklin
			# but that doesn't fix the import issue.
			raise RuntimeError("It seems __init__ has not run. Did you try to un/pickle the Frame? Pickling is not supported.")
			# self._N = len(next(iter(self.values()))[0])
		return self._N
	@property
	def shape(self):
		# required for sklearn.model_selection.cross_val_predict
		return (self._N,) # this is the only assertable common denominator
	def __len__(self): return self.N
	def __getitem__(self, selector):
		# numpy doesn't return copies of array slices and neither should this method
		if isinstance(selector, str): # dictionary key
			return super(self.__class__, self).__getitem__(selector)
		elif isinstance(selector, list): # list of dictionary keys
			if set(selector) <= set(self.keys()):
				items = ((k, self[k]) for k in selector)
				return self.__class__(items)
			else:
				missing = map(str, set(selector) - set(self.keys()))
				raise KeyError("{}".format("', '".join(missing)))
		elif isinstance(selector, int): # single datapoint -> select as list
			items = ((k, A[selector:selector+1]) for k,A in self.items())
			return self.__class__(items)
		else: # slice or numpy array
			items = ((k, A[selector]) for k,A in self.items())
			return self.__class__(items)
	def __str__(self):
		return "{}({})".format(self.__class__.__name__, super(self.__class__, self).__str__())
	def __setitem__(self, selector, value):
		if isinstance(selector, str):
			if selector not in self:
				if len(value) == self.N:
					return super(self.__class__, self).__setitem__(selector, value)
				else:
					raise ValueError("New array '{}' must have axis 0 of size {}, which was {}.".format(selector, self.N, len(value)))
			elif len(value) == len(self[selector]):
				return super(self.__class__, self).__setitem__(selector, value)
			else:
				raise ValueError("Array length must remain the same ({}), but was {} for key '{}'.".format(len(self[selector]), len(value), selector))
		elif isinstance(value, self.__class__):
			your_keys, my_keys = set(value.keys()), set(self.keys())
			additional_keys = your_keys - my_keys
			# when assigning to an array subrange, it doesn't make sense to add new arrays to the dictionary
			if len(additional_keys) == 0:
				for k in your_keys:
					self[k][selector] = value[k]
			else:
				missing = map(str, additional_keys)
				raise KeyError("Can only assign to existing key(s), but {} object has no key(s) '{}'".format(self.__class__.__name__, "', '".join(missing)))
	def copy(self):
		items = ((k, np.copy(A)) for k,A in self.items())
		return self.__class__(items)

def fit_shapes(Y, *pr):
	Y, *pr = [np.squeeze(A) for A in (Y,)+pr]
	dimsY, *dims = [len(A.shape) for A in [Y]+pr]
	pr = [A.reshape(Y.shape) if dimsY != d else A for A,d in zip(pr, dims)]
	return [Y] + pr

def zscore(alpha, df=0):
	if df == 0: # assume infty
		return sp.stats.norm.ppf(alpha)
	elif df > 0: # num_members for bootstrapped models
		return sp.stats.t.ppf(alpha, df)

def rhzscore(alpha, df=0):
	return zscore((1 + alpha)/2, df=df)

def rmse(Y, prY, force=False):
	Y, prY = map(np.squeeze, (Y, prY))
	if max(len(A.shape) for A in (Y, prY)) > 2:
		print("WARNING:   Y has shape", Y.shape)
		print("WARNING: prY has shape", prY.shape)
		if not force: return
		else: print("use force=True to still calc the rmse, but it could take long and exhaust your RAM")
	return ((np.squeeze(Y) - np.squeeze(prY))**2).mean()**0.5

def plt_stop_interaction():
	# stops interaction with interactive plots in jupyter
	# (like clicking the shutdown button on the top right of interactive igures)
	# solution curtosy of https://www.reddit.com/r/IPython/comments/47tct0/stop_interaction_in_jupyter_plot_programmatically/
	# github issue: https://github.com/matplotlib/matplotlib/issues/6071
	# alternatively simply use %matpotlib inline in the cell
	pp.gcf().canvas.draw()
	sleep(0.1)
	pp.close()

def exp_lr(start, stop, epochs, floor=None):
	'''
	creates an exponentially decaying learning rate generator
	  start: starting learning rate
	  stop: learning rate after `steps` iterations
	  epochs:
	    number of epochs to reach `stop` learning rate,
	returns
	  learning rate generator function with parameters
	    n_batches_in_epoch, global_step
	'''
	# keeping state to save computation cycles in such a trivial function was a bad idea
	def lr(batches_in_epoch, global_step):
		has_floor = floor is not None
		steps = batches_in_epoch * epochs
		decay = np.exp(np.log(stop/start)/steps)
		r = start * decay**global_step
		if has_floor and r <= floor:
			return floor
		return r
	return lr

def cycle_lr(low, high, epochs, decay_steps, decay="half", gamma=0.9997):
	"""
	cycles learning rates in range(low, high, batch_steps) symmetrically
	see smith2017 "Cyclical Learning Rates for Training Neural Networks"
	  low: minimum learning rate
	  high: maximum learning rate
	  epochs: duration of half a cycle, 2-10 epochs recommended
		decay_steps: n_batch_iterations, see gamma
	  decay: None, "halving" or "exponential"
		"None" disables decay
		"half" halves the high-low difference each cycle
		"exp" decays both bounds exponentially, see gamma
	  gamma: used for exponential dacay like [low, high] * gamma**(iteration/decay_steps)
	returns
	  learning rate generator function with parameters
	    n_batches_in_epoch, global_step
	"""
	if decay is None or decay == "None": # triangular
		def f(batches_in_epoch, i):
			cycle = np.floor(1 + i/(2*batches_in_epoch*epochs))
			x = np.abs(i/(batches_in_epoch*epochs) - 2*cycle + 1)
			lr = low + (high - low)*max(0.0, (1-x))
			return lr
	elif decay == "half": # triangular2
		def f(batches_in_epoch, i):
			cycle = np.floor(1 + i/(2*batches_in_epoch*epochs))
			x = np.abs(i/(batches_in_epoch*epochs) - 2*cycle + 1)
			lr = low + (high - low)/(2**cycle)*max(0.0, (1-x))
			return lr
	elif decay == "exp": # exp_range(gamma)
		def f(batches_in_epoch, i):
			cycle = np.floor(1 + i/(2*batches_in_epoch*epochs))
			x = np.abs(i/(batches_in_epoch*epochs) - 2*cycle + 1)
			decayed_low = low * gamma**(i/decay_steps)
			decayed_high = high * gamma**(i/decay_steps)
			lr = decayed_low + (decayed_high - decayed_low)*max(0.0, (1-x))
			return lr
	return f

# https://stackoverflow.com/questions/15792552/numpy-scipy-equivalent-of-r-ecdfxx-function
def ecdf(X, interpolate=True):
	X = np.sort(X)
	Y = np.arange(1, len(X)+1)/float(len(X))
	if not interpolate:
		return X, Y
	else:
		cdf = sp.interpolate.interp1d(X, Y)
		inv_cdf = sp.interpolate.interp1d(Y, X)
		return cdf, inv_cdf

def epdf(Y, *args, bins=100, **kwargs):
	# sometimes a scatterplot of this gives a better idea of the distribution 
	# than a barplot or cdf
	# use like: epdf(Y, "k,", bins=100)
	Y = np.sort(Y)
	L = len(Y)
	X = np.linspace(Y[0], Y[-1], bins+1)
	Y = (Y[:,None] <= X[None,:]).sum(axis=0)
	dx = (X[1] - X[0])/2 # uniform
	dY = (Y[1:] - Y[:-1])/L
	pp.plot(X[:-1] + dx, dY, *args, **kwargs)

class LossGraph:
	def __init__(self, window=9, validation_interval=50):
		self.trI, self.trL = [], [] # training
		self.vaI, self.vaL = [], [] # validation
		self.fig, self.ax = pp.subplots(1,1)
		self.window = validation_interval*window
		self.validation_interval = validation_interval
		self.training_interval = max(1, validation_interval//3)

	def _apply(self):
		self.ax.relim()
		self.ax.autoscale_view()
		self.fig.canvas.draw()
		
	def draw(self, i, tr_loss, va_loss):
		self.trI += [i]
		self.trL += [tr_loss]
		if va_loss is not None:
			self.vaI += [i]
			self.vaL += [va_loss]
		if i % self.training_interval == 0:
			if self.ax.lines:
				tr_line, *va_lines = self.ax.lines
				trR = self.trI[-self.window:]
				tr_line.set_xdata(self.trI[-self.window:])
				tr_line.set_ydata(self.trL[-self.window:])
				for va_line in va_lines:
					va_line.set_xdata(self.vaI[-self.window//self.validation_interval:])
					va_line.set_ydata(self.vaL[-self.window//self.validation_interval:])
			else:
				self.ax.plot(self.trI, self.trL, ".", color="royalblue", ms=1)
				self.ax.plot(self.vaI, self.vaL, "-", color="black", lw=0.5)
				self.ax.plot(self.vaI, self.vaL, "v", color="black", ms=4)
			self._apply()

class Timer:
	def __init__(self, duration):
		if isinstance(duration, int):
			self.duration_is_timelimit = False
			self.duration = duration
		elif isinstance(duration, float):
			self.duration_is_timelimit = True
			self.duration = duration * 60 * 60 # time in seconds
		else:
			raise ValueError('duration must be int (iterations) or float (hours)')
		self.i = 0
		self.start = time()
		print()
	def tick(self):
		self.i += 1
		# print some extra spaces at the end to overwrite trailing characters, if
		# the printed line is shorter than the line that's printed over
		print("\r"+self.string(), end="  ")
	def string(self):
		t = time() - self.start
		t_per_i = t/self.i
		if self.duration_is_timelimit:
			t_remain = self.duration - t
			n_iterations = int(self.duration/t_per_i)
		else: # duration is max. iterations number
			t_remain = t_per_i * (self.duration - self.i)
			n_iterations = self.duration
		s = t%60
		m = (t/60)%60
		h = t//3600
		elapsed = "%02i:%02i:%02i"%(h,m,s)
		s = t_remain%60
		m = (t_remain/60)%60
		h = t_remain//3600
		remain = "-%02i:%02i:%02i"%(h,m,s)
		return "%s %s %i/%i"%(elapsed, remain, self.i, n_iterations)

def logdir(name):
	now = datetime.now()
	run_time = now.strftime("%m%d-%H%M")
	name = re.subn(r'@', run_time, name)[0] + "/"
	logdir = "logs/" + name
	return logdir, run_time

def fig(x, y):
	pp.figure(figsize=(x,y))

def mulsumk(A, B):
	"""
	DONT USE THIS, PERFORMANCE SUCKS and it uses as much memory as if you used tf.tile()
	makes me kind of doubt the whole einsum hype
	broadcast-multiplies A (say of shape (aijmk) and B (say of shape (ijkn)) and sums over axis k:
	 aijmk_ (A, _:=newaxis)
	*_ij_kn (B)
	 ------ sum over k
	 aijmn
	"""
	A = tf.expand_dims(A, axis=-1)
	B = tf.expand_dims(B, axis=-3)
	return tf.reduce_sum(A*B, axis=-2)

