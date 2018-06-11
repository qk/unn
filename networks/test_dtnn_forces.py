import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class DTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, trainer="adam", momentum=0.9, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			# D = tf.placeholder(tf.float32, shape=[None, a, a, d], name="D") # input (interatomic distances)
			R = tf.placeholder(tf.float32, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			# S = tf.placeholder(tf.float32, shape=[None, a, s], name="S") # input (atom representation selection matrices)
			E = tf.placeholder(tf.float32, shape=[None, 1], name="E") # output (energies per molecule)
			F = tf.placeholder(tf.float32, shape=[None, None, 3], name="F") # output (forces per atom)
			lr = tf.placeholder(tf.float32, name='learning_rate')
			keep = tf.placeholder(tf.float32, name='keep_probability')
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
					F: sample["F"]
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
			with tf.variable_scope("interaction", initializer=self._inits.uniform):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f])
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.get_variable("Wfc", [f,c])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])

				DW = tf.matmul(D, tf.tile(Wdf[None,None,:,:], [q,a,1,1])) + b2
				# DW = DW/tf.nn.moments(DW, axes=(0,1))[1]
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
				prE = tf.reduce_sum(Ei, axis=1) # (?,1)
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				uf, of = tr_mu_sigma["F"]
				uR, oR = tr_mu_sigma["R"]
				ue = self.const("meanE", ue)
				oe = self.const("stdE", oe)
				uf = self.const("meanF", uf)
				of = self.const("stdF", of)
				uR = self.const("meanR", uR)
				oR = self.const("stdR", oR)

			# for var in tf.trainable_variables():
				# print(var)
				
			# loss
			dE = (E-ue)/oe - prE
			prF = -tf.gradients((prE*oe), [R])[0]
			# tf.summary.histogram("prF", prF)
			print(prF.shape, 'prF')
			dF = (F - prF)
			
			# loss = tf.reduce_mean(tf.square(dF))
			loss = tf.reduce_mean(tf.square(dE))
			mae = tf.reduce_mean(tf.abs((F-uf) - prF*of))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)

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
