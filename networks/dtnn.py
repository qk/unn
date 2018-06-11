import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class DTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, trainer="adam", momentum=0.9, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			# D = tf.placeholder(tf.float64, shape=[None, a, a, d], name="D") # input (interatomic distances)
			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			# S = tf.placeholder(tf.float64, shape=[None, a, s], name="S") # input (atom representation selection matrices)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			# F = tf.placeholder(tf.float64, shape=[None, a, 3], name="F") # output (forces per atom)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			is_training = tf.placeholder(tf.bool, name='is_training')
			a = tf.shape(R)[1]

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
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
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				# C = self.matmul(S, A)
				C = tf.gather(A, Z)
				# tf.summary.image("A", tf.reshape(A, [1,-1,c,1]))
			with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f])
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.get_variable("Wfc", [f,c])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])

				DW = self.matmul(D, Wdf) + b2
				# DW = DW/tf.nn.moments(DW, axes=(0,1))[1]
				for t in range(3):
					CW = self.matmul(C, Wcf) + b1
					V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=2) # (qic)
			with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [c, 15])
				b = tf.get_variable("b", [15], initializer=self._inits.zeros)
				Oi = tanh(self.matmul(C, W) + b)
			with tf.variable_scope("energies", initializer=self._inits.zeros, dtype=tf.float64):
				W = tf.get_variable("W", [15,1])
				b = tf.get_variable("b", [1])
				Ei = self.matmul(Oi, W) + b
				prE = tf.reduce_sum(Ei, axis=1) # (?,1)
			with tf.variable_scope("scaling", dtype=tf.float64):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("stdE", oe, tf.float64)

			# for var in tf.trainable_variables():
				# print(var)

			# loss
			dE = (E-ue)/oe - prE
			
			loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			step = tf.Variable(0, name='global_step', trainable=False)
			if trainer == "momentum":
			# lr = tf.train.exponential_decay(lr, step, int(100e3), 0.95)
				optimizer = tf.train.MomentumOptimizer(lr, momentum)
			elif trainer == "adam":
				optimizer = tf.train.AdamOptimizer(lr)
			else:
				raise ValueError("no such trainer '{}', choose 'adam' or 'momentum'".format(trainer))
			train = optimizer.minimize(loss, global_step=step)
	
			# summary ops
			# tf.summary.scalar("loss", loss)
			# tf.summary.scalar("rmse", rmse)
			# tf.summary.scalar("mae", mae)
			# for tname,t in [("prE", prE), ("Oi", Oi), ("C", C), ("V", V), ("CW", CW), ("DW", DW), ("Wfc", Wfc), ("Wcf", Wcf), ("A", A)]:
				# tf.summary.histogram(tname, t)

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
