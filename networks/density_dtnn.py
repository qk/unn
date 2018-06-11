import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class DensityDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
					R: sample["R"],
					Z: sample["Z"],
					E: sample["E"][:,None],
				}

			with tf.variable_scope("preprocessing"):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=mu_max
				)
				print(D.dtype, "D dtype")
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				C = tf.gather(A, Z)

			with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f], dtype=tf.float64)
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.get_variable("Wfc", [f,c])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])
				DW = self.matmul(D, Wdf) + b2
				for t in range(3):
					CW = self.matmul(C, Wcf) + b1
					V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=2) # (qic)

			with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [c, 15])
				b = tf.get_variable("b", [15], initializer=self._inits.zeros)
				Oi = tanh(self.matmul(C, W) + b) # (q,i,15)

			ys = []
			for scopename in ["mean", "variance"]:
				with tf.variable_scope(scopename):
					with tf.variable_scope("out2", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b
						y = tf.reduce_sum(yi, axis=1) # u:mu
						ys += [y]
			prE, ls2 = ys
			prec = tf.exp(-ls2) # 1/sigma**2
			prU = 1./prec # sigma**2

			with tf.variable_scope("scaling", dtype=tf.float64):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("muE", ue, tf.float64)
				oe = self.const("sigmaE", oe, tf.float64)
				
			# loss
			dE = (E-ue)/oe - prE
			negative_logl = ls2 + prec*dE*dE
			
			loss = tf.reduce_mean(negative_logl)
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
				targets = dict(E=prE*oe + ue, U=tf.sqrt(prU)*oe),
				targets_raw = dict(E=prE, U=tf.sqrt(prU))
			)


class FullDensityDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
					epoch: kw.get('epoch', 3000),
					R: sample["R"],
					Z: sample["Z"],
					E: sample["E"][:,None],
				}

			with tf.variable_scope("preprocessing", dtype=tf.float64):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=mu_max
				)
				print(D.dtype, "D dtype")
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				AZ = tf.gather(A, Z)

			ys = []
			for scopename in ["mean", "variance"]:
				C = AZ
				with tf.variable_scope(scopename):
					with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
						Wcf = tf.get_variable("Wcf", [c,f])
						b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
						Wdf = tf.get_variable("Wdf", [d,f], dtype=tf.float64)
						b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
						Wfc = tf.get_variable("Wfc", [f,c])
						diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
						diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])
						DW = self.matmul(D, Wdf) + b2
						for t in range(3):
							CW = self.matmul(C, Wcf) + b1
							V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
							V = V*diag_zeros
							C += tf.reduce_sum(tanh(V), axis=2) # (qic)
					with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [c, 15])
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W) + b) # (q,i,15)
					with tf.variable_scope("out2", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b
						y = tf.reduce_sum(yi, axis=1) # u:mu
						ys += [y]
			prE, ls2 = ys
			prec = tf.exp(-ls2) # 1/sigma**2
			prU = 1./prec # sigma**2

			with tf.variable_scope("scaling", dtype=tf.float64):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("muE", ue, tf.float64)
				oe = self.const("sigmaE", oe, tf.float64)

			# loss
			dE = (E-ue)/oe - prE
			negative_logl = tf.reduce_mean(ls2 + prec*dE*dE)
			square_loss = tf.reduce_mean(tf.square(dE))

			loss = tf.cond(
				epoch > 100,
				lambda: negative_logl,
				lambda: square_loss,
				name="loss"
			)
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
				targets = dict(E=prE*oe + ue, U=tf.sqrt(prU)*oe),
				targets_raw = dict(E=prE, U=tf.sqrt(prU))
			)

class SteppedDensityDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
					epoch: kw.get('epoch', 3000),
					R: sample["R"],
					Z: sample["Z"],
					E: sample["E"][:,None],
				}

			with tf.variable_scope("preprocessing"):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=mu_max
				)
				print(D.dtype, "D dtype")
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				AZ = tf.gather(A, Z)

			ys = []
			for scopename,T in zip(["mean", "variance"], [3,1]):
				C = AZ
				with tf.variable_scope(scopename):
					with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
						Wcf = tf.get_variable("Wcf", [c,f])
						b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
						Wdf = tf.get_variable("Wdf", [d,f], dtype=tf.float64)
						b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
						Wfc = tf.get_variable("Wfc", [f,c])
						diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
						diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])
						DW = self.matmul(D, Wdf) + b2
						for t in range(T):
							CW = self.matmul(C, Wcf) + b1
							V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
							V = V*diag_zeros
							C += tf.reduce_sum(tanh(V), axis=2) # (qic)
					with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [c, 15])
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W) + b) # (q,i,15)
					with tf.variable_scope("out2", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b
						y = tf.reduce_sum(yi, axis=1) # u:mu
						ys += [y]
			prE, ls2 = ys
			prec = tf.exp(-ls2) # 1/sigma**2
			prU = 1./prec # sigma**2

			with tf.variable_scope("scaling", dtype=tf.float64):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("muE", ue, tf.float64)
				oe = self.const("sigmaE", oe, tf.float64)

			# loss
			dE = (E-ue)/oe - prE
			negative_logl = tf.reduce_mean(ls2 + prec*dE*dE)
			square_loss = tf.reduce_mean(tf.square(dE))

			loss = tf.cond(
				epoch > 100,
				lambda: negative_logl,
				lambda: square_loss,
				name="loss"
			)
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
				targets = dict(E=prE*oe + ue, U=tf.sqrt(prU)*oe),
				targets_raw = dict(E=prE, U=tf.sqrt(prU))
			)

class HalfDensityDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			q = tf.shape(E)[0]
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
					epoch: kw.get('epoch', 3000),
					R: sample["R"],
					Z: sample["Z"],
					E: sample["E"][:,None],
				}

			with tf.variable_scope("preprocessing"):
				D, d = self._gauss_grid(self._interatomic_dists(R),
					delta_mu=0.2, sigma=0.2, mu_min=-1.0, mu_max=mu_max
				)
				print(D.dtype, "D dtype")
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				C = tf.gather(A, Z)

			with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f], dtype=tf.float64)
				b2 = tf.get_variable("b2", [f], initializer=self._inits.zeros)
				Wfc = tf.get_variable("Wfc", [f,c])
				diag_zeros = 1.0-tf.diag(tf.ones(max_atoms, dtype=tf.float64)[:a])
				diag_zeros = tf.tile(diag_zeros[:,:,None], [1,1,c])
				DW = self.matmul(D, Wdf) + b2
				for t in range(3):
					CW = self.matmul(C, Wcf) + b1
					V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=2) # (qic)

			ys = []
			for scopename,t in zip(["mean", "variance"], [3,1]):
				with tf.variable_scope(scopename):
					with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [c, 15])
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W) + b) # (q,i,15)
					with tf.variable_scope("out2", initializer=self._inits.uniform, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b
						y = tf.reduce_sum(yi, axis=1) # u:mu
						ys += [y]
			prE, ls2 = ys
			prec = tf.exp(-ls2) # 1/sigma**2
			prU = 1./prec # sigma**2

			with tf.variable_scope("scaling", dtype=tf.float64):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("muE", ue, tf.float64)
				oe = self.const("sigmaE", oe, tf.float64)

			# loss
			dE = (E-ue)/oe - prE
			negative_logl = tf.reduce_mean(ls2 + prec*dE*dE)
			square_loss = tf.reduce_mean(tf.square(dE))

			loss = tf.cond(
				epoch > 100,
				lambda: negative_logl,
				lambda: square_loss,
				name="loss"
			)
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
				targets = dict(E=prE*oe + ue, U=tf.sqrt(prU)*oe),
				targets_raw = dict(E=prE, U=tf.sqrt(prU))
			)

