import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin
from networks.dropout_ho_dtnn import q_rows, max_norm

class HeteroscedasticDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, N, mu_max=10, keep_prob=0.90, length_scale=20, n_predictions=50, tf_config=None, name=None):
		max_atoms = 20

		print(keep_prob, "(henet neural network instance keep_prob)")
		print(name, "(henet neural network instance name)")

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			is_training = tf.placeholder(tf.bool, name='is_training')
			step = tf.Variable(0, name='global_step', trainable=False)
			weight_decay = 0.0 # implies 1/tau = 0
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
					epoch: kw.get('epoch', 3000),
					R: sample["R"],
					Z: sample["Z"],
					E: sample["E"][:,None],
				}

			with tf.variable_scope("preprocessing", dtype=tf.float64):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=mu_max
				)
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal, dtype=tf.float64)
				AZ = tf.gather(A, Z)

			ys = []
			for scopename in ["mean", "variance"]:
				C = AZ
				with tf.variable_scope(scopename):
					with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
						Wcf = tf.get_variable("Wcf", [c,f])
						b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
						Wdf = tf.get_variable("Wdf", [d,f])
						b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
						Wfc = tf.get_variable("Wfc", [f,c])
						diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
						diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])

						DW = self.matmul(D, Wdf, name="DW") + b2
						for t in range(3):
							CW = self.matmul(C, Wcf, name="CW") + b1
							V = self.matmul(CW[:,None,...] * DW, Wfc, name="V") # (qijc)
							V = V*diag_zeros
							C += tf.reduce_sum(tanh(V), axis=2) # (qic)

					# DROPOUT VARIANCE STARTS HERE
					if scopename == "mean":
						n = tf.cond(is_training, lambda:1, lambda:n_predictions)
					else:
						n = 1
					C = tf.tile(C[None,...], (n,1,1,1))
					with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
						W = q_rows(tf.get_variable("W", [c,15]), keep=keep, n=n)
						W = max_norm(W, axis=1)
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W, keep_first=1, name="Oi") + b)

					with tf.variable_scope("out2", initializer=self._inits.uniform, dtype=tf.float64):
						W = q_rows(tf.get_variable("W", [15,1]), keep=keep, n=n)
						W = max_norm(W, axis=1)
						b = tf.get_variable("b", [1], initializer=self._inits.zeros)
						yi = self.matmul(Oi, W, keep_first=1, name="Ei") + b
						y = tf.reduce_sum(yi, axis=2)
						ys += [y]
			prEs, ls2s = ys
			precs = tf.exp(-ls2s) # 1/sigma**2
			sigma2s = 1./precs # sigma**2

			prE, prV = tf.nn.moments(prEs, axes=[0]) # mean, variance
			sigma2 = tf.reshape(sigma2s, [q,1]) # (batchsize,1), no dropout

			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("sigmaE", oe, tf.float64)

			trueE = (E - ue)/oe
			dEs = trueE[None,:,:] - prEs

			negative_logl = ls2s + precs*dEs*dEs # numerically more stable this way
				
			# loss
			loss = tf.cond(
				epoch > 100,
				lambda: tf.reduce_mean(negative_logl, axis=1),
				lambda: tf.reduce_mean(tf.square(dEs), axis=1),
				name="loss"
			)
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			step = tf.Variable(0, name='global_step', trainable=False)
			momentum = 0.99
			optimizer = tf.train.AdamOptimizer(lr)
			train = optimizer.minimize(loss, global_step=step)
	
			super().__init__(
				tf_config = tf_config,
				name = name,
				graph = graph,
				train_op = train,
				fill_feed = fill_feed,
				losses = dict(loss=loss, rmse=rmse, mae=mae),
				targets = dict(E=prE*oe + ue, U=tf.sqrt(prV + sigma2)*oe), # inv_tau ~ sigma**2
				targets_raw = dict(E=prE, U=tf.sqrt(prV + sigma2)),
			)
