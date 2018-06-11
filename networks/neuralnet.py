import numpy as np
import tensorflow as tf
import time
import os, sys
from matplotlib import pyplot as pp
from utils import Timer, LossGraph, logdir
from collections import defaultdict

__all__ = [
	"MolecularPreprocessingMixin",
	"NeuralNet",
]

class MolecularPreprocessingMixin:
	def _interatomic_dists(self, R):
		'''
		transforms absolute coordinates in data matrix R to pairwise distance matrices
		params:
			R - matrix of shape (n_points, n_atoms, 3)
		returns
			- pairwise distance matrix of shape (n_points, n_atoms, n_atoms)
		'''
		with tf.variable_scope('interatomic_dists'):
			dists = tf.sqrt(tf.reduce_sum(tf.square(R[:,None,:,:] - R[:,:,None,:]), axis=-1))
		return dists
	
	def _gauss_grid(self, dists, delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=10):
		'''
		expands single values on a grid of gaussians, where the grid consists of 
		centers roughly like linspace(mu_min, mu_max, (mu_max-mu_min)/delta_mu)
		params:
			dists - pairwise distance matrix of shape (n_points, n_atoms, n_atoms)
			delta_mu - space between gaussian centers
			sigma - width of gaussians (same for all)
			mu_min - minimum location of centers
			mu_max - maximum location of centers
		returns
			- expanded distance matrix of shape (n_points, n_atoms, n_atoms, n_centers)
			- n_centers (useful to get static shapes of tensors)
		'''
		with tf.variable_scope('gauss_grid'):
			num = int(np.ceil((mu_max-mu_min)/delta_mu))
			remainder = num*delta_mu - (mu_max - mu_min)
			# handling the remainder makes sure the stepsize is upheld exactly 
			mu = self.const('centers', np.linspace(
				mu_min + remainder/2,
				mu_max - remainder/2,
				num
			), dists.dtype)
			G = tf.exp(- tf.square(dists[:,:,:,None] - mu[None,None,None,:])/(2.0*sigma**2))
			G = G*tf.cast(dists >= 1e-5, G.dtype)[:,:,:,None]
		return G, num

class NeuralNet:
	# baseclass with all common methods (batch-splitting, training). does everything except construct the graph and choose the optimizer.
	class _reg:
		l1 = tf.contrib.layers.l1_regularizer
		l2 = tf.contrib.layers.l2_regularizer
	class _configs:
		useCPU = tf.ConfigProto(
			intra_op_parallelism_threads=5,
			inter_op_parallelism_threads=5,
			# allow_soft_placement=True,
			device_count = {'GPU': 0}
		)
		saveMemory = tf.ConfigProto(
			# max parallelization speedup of single op
			intra_op_parallelism_threads=4,
			# max concurrent paths in graph
			inter_op_parallelism_threads=2,
			# allow_soft_placement=True,
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
		)
		noMemoryLimit = tf.ConfigProto(
			# max parallelization speedup of single op
			intra_op_parallelism_threads=4,
			# max concurrent paths in graph
			inter_op_parallelism_threads=2,
			# allow_soft_placement=True,
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
		)
	class _inits:
		normal = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
		uniform = tf.contrib.layers.variance_scaling_initializer(uniform=True,dtype=tf.float64)
		zeros = tf.constant_initializer(0.0,dtype=tf.float64)
		ones = tf.constant_initializer(1.0,dtype=tf.float64)

	def __init__(self, fill_feed=None, graph=None, losses=None, name=None, summary=None, targets=None, targets_raw=None, tf_config=None, train_op=None):
		# this sets all attributes, that are required for this class to work

		if None in (fill_feed, graph, losses, targets, train_op):
			raise ValueError("fill_feed, graph, losses, targets, targets_on_scale, train_op must all be set.")
		if 'loss' not in losses.keys():
			raise ValueError("losses must have a 'loss=loss_op' entry. It will be used to determine the best model.")
		if not (isinstance(targets, dict) and (targets_raw is None or isinstance(targets_raw, dict))):
			raise ValueError("targets (and targets_raw if provided) must be dicts, f.i. targets=dict(Y=prY*std + mean)")
		if not callable(fill_feed):
			raise ValueError("fill_feed must be a callable.")

		self.name = name or self.__class__.__name__ # for logs and model persistence, may be overridden by fit() or predict*()
		self.graph = graph # tensorflow computation graph
		self.historyfile = None # training-history filename, value set in self.fit()
		self._initialize = True # wether variable initialization is needed (False for saved models)

		with self.graph.as_default():
			# for k,v in losses.item():
				# tf.summary.scalar(k,v)
			self._saver = tf.train.Saver() # graph saver, for model persistence
			self._summary = tf.summary.merge_all() # tensorboard summary op
			self._train = train_op # optimizer.minimize(loss) op
			self._losses = losses # dictionary of (scorename, score_op) pairs
			self._fill_feed = fill_feed # function to input datapoints into the graph
			self._targets_raw = dict((k,tf.squeeze(v)) for k,v in targets_raw.items()) # dictionary of (name, op) pairs for normalized prediction targets
			self._targets = dict((k,tf.squeeze(v)) for k,v in targets.items()) # same as targets_raw, but with normalization undone (these are the predictions you'd want most of the time)
			self._variable_initializer = tf.global_variables_initializer()
			self._session = tf.Session(graph=self.graph, config=tf_config or self._configs.saveMemory)

	def load(self, filename=None, verbose=True):
		''' restores session from file "./trained_models/{filename}/session" '''
		if filename is None:
			filename = os.path.join("trained_models", self.name)
		filename = os.path.join(filename, "session")
		if verbose: print("restoring session from", filename)
		_ = self._saver.restore(self._session, filename)
		self._initialize = False

	def save(self, filename=None, verbose=True):
		''' saves session in file "./trained_models/{filename}/session" '''
		if filename is None:
			filename = os.path.join("trained_models/", self.name)
		if not os.path.exists(filename):
			os.makedirs(filename)
		filename = os.path.join(filename, "session")
		if verbose: print("saving model as", filename)
		self._saver.save(self._session, filename)

	def clear_save(self, filename=None, verbose=True):
		min_savepath = "trained_models/{}/".format(filename)
		if os.path.exists(min_savepath) and not os.path.isfile(min_savepath):
			# clear all saved model data (probably unnecessary)
			for f in os.listdir(min_savepath):
				try:
					os.remove(os.path.join(min_savepath, f))
				except Exception as e:
					print('couldn\'t clear save directory, because of following error:')
					print(e)
					print('continuing...')

	def const(self, name, v, dtype=None):
		'''
		creates non-trainable variables, that are stored in the graph session.
		useful to persist training set scaling parameters and other fixed values in the graph.
		DTYPE will always be tf.float32.
		params:
		  name -- name of the variable
			v -- value
		'''
		shape = ()
		dtype = tf.float32 if dtype is None else dtype
		if hasattr(v, 'shape'): shape = v.shape
		elif isinstance(v, (list, tuple)): shape = (len(v),)
		return tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(v), trainable=False)

	def matmul(self, A, B, keep_first=0, name=None):
		shapeA = tf.shape(A) # (a,b,c,...,m,n)
		shapeB = tf.shape(B) # (a,b,c,    n,z)
		# fold axes of A that cannot be aligned with B
		# s.t. A has new shape (a,b,c,-1,n)
		folded_shapeA = tf.concat([shapeA[:keep_first], [-1,shapeA[-1]]], 0)
		# unfold folded axes of A in the result,
		# s.t. result has shape (a,b,c,...,m,z)
		folded_name = "folded_matmul"+("_of_"+name if name is not None else "")
		unfolded_result_shape = tf.concat([
				shapeA[:keep_first],
				shapeA[keep_first:-1],
				[shapeB[-1]]
			], 0)
		return tf.reshape(
			tf.matmul(tf.reshape(A, folded_shapeA), B, name=folded_name),
			unfolded_result_shape,
			name=name
		)
	
	def _batch_mean(self, sess, ops, D):
		# conserves gpu memory at slight performance and arbitrary precision cost.
		k = list(D[0].keys())[0] # any key
		sizes = np.array([len(batch[k]) for batch in D])
		N = sizes.sum()
		# attempt to compute the mean properly, but since it's unclear which value 
		# each op returns, this won't be 100% accurate in every case (f.i. with 
		# rmse). for accurate numbers keep the va set size below 2k points or don't 
		# apply non-linearities after computing the batch mean in the loss-ops.
		return [
			(np.array(
				[sess.run(op, feed_dict=self._fill_feed(batch)) for batch in D]
			)*sizes).sum()/N
			for op in ops
		]

	def fit(self, trD, vaD, batch_size=200, epochs=30, va_batch_size=2000, name=None, silent=False, lr=1e-3, plot_loss='loss', shuffle=True, stop_at=None, restart_check=None):
		name = name or self.name
		if isinstance(lr, (int, float)): 
			const_lr = lr
			lr = lambda *args: float(const_lr)
		summarize = self._summary is not None # and None not in self._summary (when did i need this?)

		trN = len(next(iter(trD.values())))
		vaN = len(next(iter(vaD.values())))
		trI = np.arange(trN)
		batches = trN//batch_size
		vaChunks = self._split(vaD, va_batch_size) if vaN > va_batch_size else [vaD]
		validation_interval = batches//1 + 1 # added 1 ensures we don't validate just before an epoch ends
		save_interval = 5 # minimum distance in epochs between save checkpoints

		# always save the network with minimum validation loss
		va_loss = np.infty
		min_va_loss = np.infty
		min_epoch = 0
		min_savepath = "trained_models/{}/".format(name)
		print(min_savepath)
		if os.path.exists(os.path.join(min_savepath, "is.trained")):
			print("Found", os.path.join(min_savepath, "is.trained"), 'indicator file, training is manually skipped.')
			return self
		self.clear_save(name)
		reset_epochs = defaultdict(int) # counts how many times training failed and returned to a specific minimum epoch

		# logging
		logfolder, run_time = logdir("@-%s" % self.name)
		if not silent:
			self.historyfile = "trained_models/{}/{}-T.npz".format(self.name, run_time)
			print("history will be written to", self.historyfile)
			T = []
			clock = Timer(epochs)
			plot = LossGraph(validation_interval=validation_interval)
			pp.ylabel(plot_loss)
			pp.xlabel("batch iteration i of %i w/ batch size %i" % (batches, batch_size))
		# one might argue if summary should only be written when not in silent mode
		if summarize:
			summary_writer = tf.summary.FileWriter(logfolder, graph=self.graph)
			print("logs saved in", logfolder)
		else:
			print("no summary ops specified, no tensorboard logs will be written")

		loss_op = self._losses['loss']
		score_names, score_ops = map(
			list,
			zip(*[(k,v)
				for k,v in sorted(self._losses.items(), key=lambda e: e[0])
				if k != 'loss']) # zip(*list_of_tuples) unzips the list of tuples
		)
		tr_ops = [self._train, loss_op] + score_ops
		if not silent and plot_loss not in self._losses.keys():
			raise ValueError('No such loss: "{}". Choose from {}'.format(plot_loss, self._losses.keys()))
		plot_loss_op = self._losses[plot_loss]
		plot_loss_i = 0 if plot_loss not in score_names else 1+score_names.index(plot_loss)
		if stop_at is not None:
			k,stop_loss_threshold = next(iter(stop_at.items()))
			stop_loss_op = self._losses[k]
		if restart_check is not None:
			restart_loss_op = self._losses[restart_check[0]]
			restart_check_threshold = restart_check[1]
			restart_check_epoch = restart_check[2]

		t_start = time.time()
		stop = False
		# mean_guess_rmse = ((vaD["T"] - vaD["T"].mean())**2).mean()**0.5
		if self._initialize:
			self._session.run(self._variable_initializer)
		epoch = -1
		while (epochs == -1 or (epochs > 0 and epoch+1 < epochs)) and not stop:
			epoch += 1
			trIndexChunks = self.shuffle_and_split(trI, batches, shuffle)
			i = -1
			while i+1 < batches and not stop:
				i += 1

				learning_rate = lr(batches, epoch*batches + i)
				tr_dict = self._fill_feed(self.sample(trD, trIndexChunks, i), lr=learning_rate, is_training=True, epoch=epoch)
				_, *tr_scores = self._session.run(tr_ops, feed_dict=tr_dict)
				tr_loss = tr_scores[0]
				if summarize:
					summary = self._session.run([self._summary], feed_dict=tr_dict)[0]
					summary_writer.add_summary(summary, i + epoch*batches)
				if not silent:
					plot.draw(i+epoch*batches, tr_scores[plot_loss_i], None)

			if stop_at is not None:
				va_loss, stop_loss, *_ = self._batch_mean(self._session, [loss_op, stop_loss_op], vaChunks)
				stop = stop_loss < stop_loss_threshold
			else:
				va_loss, *_ = self._batch_mean(self._session, [loss_op], vaChunks)
			if np.isnan(va_loss):
				# raise ValueError("validation loss is NaN (epoch {})".format(epoch))
				print("validation loss is NaN (epoch {})".format(epoch), file=sys.stderr)
				if min_epoch > 0:
					print("resetting to epoch {}".format(min_epoch), file=sys.stderr)
					self.load(filename=min_savepath, verbose=not silent)
					epoch = min_epoch
					reset_epochs[epoch] += 1
					if reset_epochs[epoch] > 3:
						raise ValueError("validation loss is NaN. tried to recover but failed.")
					else:
						continue
			if va_loss < min_va_loss and epoch - min_epoch > save_interval:
				# print("saving intermediate model at epoch %i with loss %f" % (epoch, va_loss))
				while min_epoch != epoch: # repeat save op until it succeeds
					try:
						self.save(filename=min_savepath, verbose=False)
						min_va_loss = va_loss
						min_epoch = epoch
					except Exception as e:
						print(e, "\nsaving failed, waiting 5m before retry")
						time.sleep(60*5)
			if restart_check is not None and epoch == restart_check_epoch:
				restart_loss = self._batch_mean(self._session, [restart_loss_op], vaChunks)[0]
				print("restart check with value {} < {} threshold".format(
					restart_loss, restart_check_threshold
				))
				if restart_loss > restart_check_threshold:
					print("check failed, restarting...")
					epoch = -1
					self._session.run(self._variable_initializer)
				else:
					print("check passed")
			if not silent:
				va_scores = [va_loss] + self._batch_mean(self._session, score_ops, vaChunks)
				named_va_scores = tuple(zip(['loss'] + score_names, va_scores))
				T += [[i + epoch*batches, epoch, named_va_scores]]
				pp.title("epoch {}/{} lr {:1.1e}\nva {}".format(
					epoch+1,
					"inf" if epochs == -1 else str(epochs),
					learning_rate,
					" ".join([str(e) for tup in named_va_scores for e in tup]))
				)
				tr_score = tr_scores[plot_loss_i]
				va_score = va_scores[plot_loss_i]
				plot.draw(i+epoch*batches, tr_score, va_score)
				clock.tick()

		if not silent: # save training history
			# this looks convoluted, but all it does is convert T to a dict
			# of form {valuename: valuearray} and saves it in `historyfile`
			T = list(zip(*T)) # zip(*tuples) unzips the tuples (it's like a transpose)
			T[2] = zip(*T[2])
			T_scores = [(k[0],np.array(v)) for k,v in [zip(*named_scores) for named_scores in T[2]]]
			np.savez(self.historyfile, iteration=T[0], epoch=T[1], **dict(T_scores))

		if not silent:
			print("\nbest epoch", min_epoch, "with va_loss", min_va_loss)
			pp.close('all')

		# return best encountered model
		self.load(filename=min_savepath, verbose=not silent)

		return self

	def _predict(self, D, ops_dict):
		if self._initialize:
			raise RuntimeError('Session is uninitialized.')
		if self._session._closed:
			raise RuntimeError('Session has already been closed.')
		chunks = self._split(D, 200)
		targets = dict()
		for k,prY in ops_dict.items():
			targets[k] = np.concatenate([self._session.run([prY], feed_dict=self._fill_feed(batch))[0] for batch in chunks]).squeeze()
		return targets

	def predict_raw(self, D):
		'''
		params
			D: either a list of batch sample dictionaries to pass to feed() individually, or one large input data dictionary to be split up automatically
		returns
			dictionary of raw (normalized for the neural network) predicted values
			f.i. {'E':np.array(...), 'U':np.array(...)}
		'''
		if not hasattr(self, '_targets_raw'):
			raise RuntimeError('{} did not define raw targets'.format(self.name))
		return self._predict(D, self._targets_raw)

	def predict(self, D):
		'''
		params
			D: either a list of batch sample dictionaries to pass to feed() individually, or one large input data dictionary to be split up automatically
		returns
			dictionary of predicted values on original scale
			f.i. {'E':np.array(...), 'U':np.array(...)}
		'''
		return self._predict(D, self._targets)

	def lr_range_test(self, D, low=0.0, high=0.5, epochs=4, batch_size=200, plot=True, shuffle=True, loss='loss'):
		N = len(D[list(D.keys())[0]])
		batches = N//batch_size
		indexChunks = self.shuffle_and_split(np.arange(N), batches, shuffle)

		J = [] # losses
		LR = np.linspace(low, high, epochs*batches)

		clock = Timer(epochs)
		self._session.run([self._variable_initializer])
		loss_op = self._losses[loss]
		for epoch in range(epochs):
			for i in range(batches):

				lr = LR[epoch*batches + i]
				tr_dict = self._fill_feed(self.sample(D, indexChunks, i), lr=lr, is_training=True)
				_, tr_loss = self._session.run([self._train, loss_op], feed_dict=tr_dict)
				if np.isnan(tr_loss):
					print("\rWARNING: loss is NaN for lr", lr, end="")
					break
				J += [tr_loss]

			clock.tick()
			indexChunks = self.shuffle_and_split(np.arange(N), batches, shuffle)
		J = np.array(J)
		LR = LR[:len(J)]

		if plot:
			pp.plot(LR, J, "k,")
			jmin = J.argmin()
			pp.gca().set_xticks([LR[jmin]], minor=True)
			pp.gca().set_xticklabels(["\n{:e}".format(LR[jmin])], minor=True)
			pp.grid(which='minor')
			pp.grid()

		return LR, J

	def _split(self, D, chunk_size): # split prediction input data dict D into chunks to fit them into gpu memory
		if isinstance(D, list):
			return D
		elif len(D) <= chunk_size:
			return [D]
		any_key = list(D.keys())[0]
		N = D[any_key].shape[0]
		num_chunks = N//chunk_size
		chunks = [{} for _ in range(num_chunks)]
		for k in D.keys():
			k_chunks = np.array_split(D[k], num_chunks, axis=0)
			for i,feed in enumerate(chunks):
				feed[k] = k_chunks[i]
		return chunks

	def shuffle_and_split(self, trI, batches, shuffle): # shuffle training set indices (between epochs)
		if shuffle:
			return np.array_split(np.random.permutation(trI), batches)
		else:
			# come on man, they should be shuffled.
			print("Datapoints aren't shuffled between epochs.")
			return np.array_split(trI, batches)

	def sample(self, trD, trIndexChunks, i):
		# could be done in self._fill_feed, but allowing for this overhead here gives cleaner code overall
		feed = {}
		for k in trD.keys():
			feed[k] = trD[k][trIndexChunks[i]]
		return feed

	def close(self):
		# unfortunately this dosen't seem to clear gpu memory when called from within a jupyter notebook
		self._session.close()
