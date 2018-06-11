import numpy as np
import tensorflow as tf
from networks.neuralnet import *
from utils import Frame

class BatchNormalizationVarianceDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, n_predictions=100, prediction_batch_size=25, tf_config=None, name=None):
		# _,_,s = point['selection_matrices'].shape
		self._n_predictions = n_predictions
		self._n_batches = n_predictions
		self._prediction_batch_size = prediction_batch_size
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			# D = tf.placeholder(tf.float32, shape=[None, None, None, None], name="D") # input (interatomic distances)
			R = tf.placeholder(tf.float32, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			# S = tf.placeholder(tf.float32, shape=[None, a, s], name="S") # input (atom representation selection matrices)
			E = tf.placeholder(tf.float32, shape=[None, 1], name="E") # output (energies per molecule)
			# F = tf.placeholder(tf.float32, shape=[None, a, 3], name="F") # output (forces per atom)
			lr = tf.placeholder(tf.float32, name='learning_rate')
			keep = tf.placeholder(tf.float32, name='keep_probability')
			is_training = tf.placeholder(tf.bool, name='is_training')
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
					is_training: kw.get('is_training', False),
					D: sample["D"],
					Z: sample["Z"],
					# S: sample["selection_matrices"].astype(float),
					E: sample["E"][:,None],
					# F: sample["F"]
				}

			with tf.variable_scope("preprocessing"):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=20.0
				)
			with tf.variable_scope("init", initializer=self._inits.normal):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				# C = dot(S, A)
				C = tf.gather(A, Z)
				# tf.summary.image("A", tf.reshape(A, [1,-1,c,1]))
			with tf.variable_scope("batch_normalization"):
				print(D.shape)
				mean, var = tf.nn.moments(D, 0)
				scale = self.const("scale", 1)
				offset = self.const("offset", 0)
				# D = tf.nn.fused_batch_norm(D,  mean=mean, variance=var)
				# TODO: update like in simplenetwork
				normD = tf.cond(
					is_training,
					lambda: tf.nn.batch_normalization(D, mean, var, offset, scale, 1e-5), # True
					lambda: D # False, use already normalized D as is
				)
				print(normD.shape)
			with tf.variable_scope("interaction", initializer=self._inits.uniform):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f])
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.get_variable("Wfc", [f,c])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])

				DW = tf.matmul(D, tf.tile(Wdf[None,None,:,:], [q,a,1,1])) + b2
				for t in range(3):
					CW = tf.matmul(C, tf.tile(Wcf[None,:,:], [q,1,1])) + b1
					V = tf.matmul(CW[:,None,:,:] * DW, tf.tile(Wfc[None,None,:,:], [q,a,1,1])) # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=2) # (qic)
			with tf.variable_scope("out1", initializer=self._inits.uniform):
				W = tf.get_variable("W", [c, 15])
				b = tf.get_variable("b", [15], initializer=self._inits.zeros)
				# Oi = tanh(dot(C, W) + b) # (q,i,15)
				Oi = tanh(tf.matmul(C, tf.tile(W[None,...], [q,1,1])) + b)
			with tf.variable_scope("energies", initializer=self._inits.uniform):
				W = tf.get_variable("W", [15,1])
				b = tf.get_variable("b", [1])
				Ei = tf.matmul(Oi, tf.tile(W[None,:,:], [q,1,1])) + b
				u = tf.reduce_sum(Ei, axis=1) # u:mu
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue)
				oe = self.const("stdE", oe)

			# loss
			prE = u # (?, 1)
			dE = (E-ue)/oe - prE
			
			loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			step = tf.Variable(0, name='global_step', trainable=False)
			# lr = tf.train.exponential_decay(lr, step, int(100e3), 0.95)
			momentum = 0.98
			# optimizer = tf.train.MomentumOptimizer(lr, momentum)
			optimizer = tf.train.AdamOptimizer(lr)
			# optimizer = tf.train.AdamOptimizer(lr, momentum)
			train = optimizer.minimize(loss, global_step=step)
	
			super().__init__(
				tf_config = tf_config,
				name = name,
				graph = graph,
				train_op = train,
				fill_feed = fill_feed,
				losses = dict(loss=loss, rmse=rmse, mae=mae),
				targets = dict(E=prE*oe + ue),
				targets_raw = dict(E=prE),
			)

	def _predict(self, D, ops_dict, is_validation_run=False):
		if self._initialize:
			raise RuntimeError('Session is uninitialized')
		if self._session._closed:
			raise RuntimeError('Session has already been closed.')

		# need the original set for D, not prefabricated batches
		if isinstance(D, list):
			keys = D[0].keys()
			data = Frame((k,np.concatenate([d[k] for d in D])) for k in keys)
		elif not isinstance(D, Frame) and isinstance(D, dict):
			data = Frame(**D)
		else:
			data = D

		prE = ops_dict["E"]
		Es = []
		I = np.arange(len(data))

		# predict once for the function value using global normalization
		# chunks = self._split(D, 1000)
		# E = np.concatenate([self._session.run([prE], feed_dict=self._fill_feed(batch, is_training=False))[0] for batch in chunks])
		# Es += [E]

		# predict many times for the uncertainty using additional batch normalization
		if not is_validation_run:
			for i in range(self._n_predictions):
				I = np.random.permutation(I)
				chunks = self._split(D[I], self._prediction_batch_size)
				E = np.concatenate([self._session.run([prE], feed_dict=self._fill_feed(batch, is_training=True))[0] for batch in chunks])
				Es += [E[np.argsort(I)]]

		return dict(
			E = np.mean(Es, axis=0),
			U = np.var(Es, axis=0),
		)

		# if not is_validation_run:
			# return dict(
				# E = Es[0], #, np.mean(Es, axis=0),
				# U = np.var(Es, axis=0),
			# )
		# else:
			# return dict(
				# E = Es[0], #, np.mean(Es, axis=0),
			# )
