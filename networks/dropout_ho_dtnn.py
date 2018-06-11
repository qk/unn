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
	selector = tf.reshape(tf.cast(tf.multinomial(tf.log([[1-keep, keep]]), n*a), tf.float64), (n,a,1))
	return selector*W[...,None,:,:]

def max_norm(W, axis=1):
	with tf.variable_scope("max_norm"):
		norm = tf.norm(W,ord=2,axis=axis,keep_dims=True)
		norm = tf.where(tf.greater(norm, 1.0), norm, tf.ones_like(norm))
		return W/norm

class HomoscedasticDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, N, mu_max=10, keep_prob=0.90, length_scale=20, n_predictions=50, tf_config=None, name=None):
		max_atoms = 20
		print(keep_prob, "(honet neural network instance keep_prob)")
		print(name, "(honet neural network instance name)")

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			# D = tf.placeholder(tf.float64, shape=[None, None, None, None], name="D") # input (interatomic distances)
			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			# S = tf.placeholder(tf.float64, shape=[None, a, s], name="S") # input (atom representation selection matrices)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			# F = tf.placeholder(tf.float64, shape=[None, a, 3], name="F") # output (forces per atom)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			is_training = tf.placeholder(tf.bool, name='is_training')
			step = tf.Variable(0, name='global_step', trainable=False)
			# weight_decay = length_scale**2*keep_prob/(2*N) # implies tau == 1
			weight_decay = 0.0 # implies 1/tau = 0
			# print(weight_decay, "l2 weight decay")
			weight_decay = self.const("weight_decay", weight_decay, tf.float64)
			n_points = self.const("n_training_points", N, tf.float64)
			scale = self.const("length_scale", length_scale, tf.float64)
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]

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

			with tf.variable_scope("preprocessing", dtype=tf.float64):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=mu_max
				)
				# print(D.shape, "D")
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				C = tf.gather(A, Z)
				# tf.summary.image("A", tf.reshape(A, [1,-1,c,1]))
				# print(C.shape, "C")
			with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f])
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.get_variable("Wfc", [f,c])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])

				# print(Wcf.shape, 'Wcf')
				# print(Wdf.shape, 'Wdf')
				# print(Wfc.shape, 'Wfc')

				DW = self.matmul(D, Wdf, name="DW") + b2
				for t in range(3):
					CW = self.matmul(C, Wcf, name="CW") + b1
					V = self.matmul(CW[:,None,...] * DW, Wfc, name="V") # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=2) # (qic)

			# DROPOUT VARIANCE STARTS HERE
			n = tf.cond(is_training, lambda:1, lambda:n_predictions)
			C = tf.tile(C[None,...], (n,1,1,1))
			with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [c,15])
				W = q_rows(W, keep=keep, n=n)
				W = max_norm(W, axis=1)
				b = tf.get_variable("b", [15], initializer=self._inits.zeros)
				Oi = tanh(self.matmul(C, W, keep_first=1, name="Oi") + b)
				# print(Oi.shape, "Oi")
			with tf.variable_scope("energies", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [15,1])
				W = q_rows(W, keep=keep, n=n)
				W = max_norm(W, axis=1)
				b = tf.get_variable("b", [1], initializer=self._inits.zeros)
				Ei = self.matmul(Oi, W, keep_first=1, name="Ei") + b
				# print(Ei.shape, "Ei")
				prEs = tf.reduce_sum(Ei, axis=2) # u:mu
				# print(prEs.shape, "prEs")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("sigmaE", oe, tf.float64)

			prE, prV = tf.nn.moments(prEs, axes=[0]) # mean, variance
			# print(E.shape, "E")
			# print(prE.shape, "prE")
			trueE = (E - ue)/oe
			dEs = trueE[None,:,:] - prEs
				
			# loss
			# if you calculate weight_decay assuming tau=1, this will be 1
			inv_tau = (2 * n_points * weight_decay)/(scale**2 * keep_prob)
			loss = tf.reduce_mean(tf.square(dEs), axis=1)
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(lr)
			train = optimizer.minimize(loss, global_step=step)

			super().__init__(
				tf_config = tf_config,
				name = name,
				graph = graph,
				train_op = train,
				fill_feed = fill_feed,
				losses = dict(loss=loss, rmse=rmse, mae=mae),
				targets = dict(E=prE*oe + ue, U=tf.sqrt(prV + inv_tau)*oe), # inv_tau ~ sigma**2
				targets_raw = dict(E=prE, U=tf.sqrt(prV + inv_tau)),
			)
