import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

def q_rows(W, keep=0.9, n=10):
	# params:
	# -------
	# W - weight matrix of shape (...,a,b)
	# keep - keep probability (fraction of rows from axis (a) to keep)
	# n - how many selection vectors to generate
	# -------
	# returns - (...,n,a,b)-matrix with random rows of axis (a) dropped
	a = tf.cast(W.shape[-2], tf.int32)
	selector = tf.reshape(
		tf.cast(tf.multinomial(tf.log([[1-keep, keep]]), n*a), tf.float32),
		(n,a,1)
	)
	# print(selector.shape, 'sel') # (n,a,1)
	# print(W.shape, 'w') # (a,b)
	# ...,n,a,1  selector
	# ...,1,a,b  W
	# ...,n,a,b  result
	return selector*W[...,None,:,:]

class HomoscedasticDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, N, keep_prob=0.95, length_scale=20, n_predictions=10, tf_config=None, name=None):
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
			step = tf.Variable(0, name='global_step', trainable=False)
			weight_decay = length_scale**2*keep_prob/(2*N) # implies tau == 1
			# print(weight_decay, "l2 weight decay")
			weight_decay = self.const("weight_decay", weight_decay)
			n_points = self.const("n_training_points", N)
			scale = self.const("length_scale", length_scale)
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]
			n = tf.cond(is_training, lambda:1, lambda:n_predictions)

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: keep_prob, # always drop
					is_training: kw.get('is_training', False),
					# D: sample["D"],
					R: sample["R"],
					Z: sample["Z"],
					# S: sample["selection_matrices"].astype(float),
					E: sample["E"][:,None],
					# F: sample["F"]
				}

			with tf.variable_scope("preprocessing"):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=10.0
				)
				D = tf.tile(D[None,...], (n,1,1,1,1))
				print(D.shape, "D")
			with tf.variable_scope("init", initializer=self._inits.normal):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				C = tf.gather(A, Z)
				# tf.summary.image("A", tf.reshape(A, [1,-1,c,1]))
				C = tf.tile(C[None,...], (n,1,1,1))
				print(C.shape, "C")
			with tf.variable_scope("interaction", initializer=self._inits.uniform):
				Wcf = tf.clip_by_norm(tf.get_variable("Wcf", [c,f]), 2, axes=[0])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.clip_by_norm(tf.get_variable("Wdf", [d,f]), 2, axes=[0])
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.clip_by_norm(tf.get_variable("Wfc", [f,c]), 2, axes=[0])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])

				print(Wcf.shape, 'Wcf')
				print(Wdf.shape, 'Wdf')
				print(Wfc.shape, 'Wfc')

				Wdf = q_rows(Wdf, keep=keep, n=n)
				DW = tf.matmul(D, tf.tile(Wdf[:,None,None,:,:], [1,q,a,1,1]), name="DW") + b2
				for t in range(3):
					qWcf = q_rows(Wcf, keep=keep, n=n)
					qWfc = q_rows(Wfc, keep=keep, n=n)
					CW = tf.matmul(C, tf.tile(qWcf[:,None,:,:], [1,q,1,1]), name="CW") + b1
					V = tf.matmul(CW[:,:,None,...] * DW, tf.tile(qWfc[:,None,None,:,:], [1,q,a,1,1]), name="V") # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=3) # (qic)
			with tf.variable_scope("out1", initializer=self._inits.uniform):
				W = q_rows(
					tf.clip_by_norm(tf.get_variable("W", [c,15]), 2, axes=[0]),
					keep=keep, n=n
				)
				b = tf.get_variable("b", [15], initializer=self._inits.zeros)
				Oi = tanh(tf.matmul(C, tf.tile(W[:,None,...], [1,q,1,1]), name="Oi") + b)
				print(Oi.shape, "Oi")
			with tf.variable_scope("energies", initializer=self._inits.uniform):
				W = q_rows(
					tf.clip_by_norm(tf.get_variable("W", [15,1]), 2, axes=[0]),
					keep=keep, n=n
				)
				b = tf.get_variable("b", [1], initializer=self._inits.zeros)
				Ei = tf.matmul(Oi, tf.tile(W[:,None,:,:], [1,q,1,1]), name="Ei") + b
				print(Ei.shape, "Ei")
				u = tf.reduce_sum(Ei, axis=2) # u:mu
				print(u.shape, "u")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue)
				oe = self.const("sigmaE", oe)

			prE, prV = tf.nn.moments(u, axes=[0]) # mean, variance
			print(E.shape, "E")
			print(prE.shape, "prE")
			trueE = (E - ue)/oe
			dE = trueE - prE # TODO: fix
				
			# loss
			# if you calculate weight_decay assuming tau=1, guess what this'll be
			inv_tau = (2 * n_points * weight_decay)/(scale**2 * keep_prob)
			loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			step = tf.Variable(0, name='global_step', trainable=False)
			momentum = 0.99
			# optimizer = tf.train.MomentumOptimizer(lr, momentum)
			# optimizer = tf.train.AdamOptimizer(learning_rate, momentum)
			optimizer = tf.train.AdamOptimizer(lr)
			train = optimizer.minimize(loss, global_step=step)
	
			# summary ops
			# tf.summary.scalar("loss", loss)
			# tf.summary.scalar("rmse", rmse)
			# tf.summary.scalar("mae", mae)

			super().__init__(
				tf_config = tf_config,
				name = name,
				graph = graph,
				train_op = train,
				fill_feed = fill_feed,
				losses = dict(loss=loss, rmse=rmse, mae=mae),
				targets = dict(E=prE*oe + ue, U=(prV + inv_tau)*oe**2), # inv_tau ~ sigma**2
				targets_raw = dict(E=prE, U=(prV + inv_tau)),
			)
