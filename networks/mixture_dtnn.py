import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class DensityMixtureDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			# S = tf.placeholder(tf.float64, shape=[None, a, s], name="S") # input (atom representation selection matrices)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			# F = tf.placeholder(tf.float64, shape=[None, a, 3], name="F") # output (forces per atom)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			step = tf.Variable(0, name='global_step', trainable=False)
			q = tf.shape(E)[0] # batch size (q, because it's printed as a question mark)
			f, c = 60, 30 # 60, 30
			a = tf.shape(R)[1]
			# q, a = 25, 12

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
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
				C = tf.gather(A, Z) # (?,12,30)
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
				for t in range(3):
					CW = self.matmul(C, Wcf) + b1
					V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
					V = V*diag_zeros
					C += tf.reduce_sum(tanh(V), axis=2) # (qic)
			with tf.variable_scope("out1", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [c, 15])
				b = tf.get_variable("b", [15], initializer=self._inits.zeros)
				# Oi = tanh(matmul(C, W) + b) # (q,i,15)
				Oi = tanh(self.matmul(C, W) + b)
			# predict means (energies), std (uncertainty) and priors (alpha)
			with tf.variable_scope("energies", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [15,1])
				b = tf.get_variable("b", [1])
				Ei = self.matmul(Oi, W) + b # (?,12,1)
				mu = tf.reshape(Ei, [-1,a])
			with tf.variable_scope("uncertainty", initializer=self._inits.uniform, dtype=tf.float64):
				W = tf.get_variable("W", [15,1])
				b = tf.get_variable("b", [1])
				ln_sigma2 = self.matmul(Oi, W) + b # (?,12,1)
				# sigma may come close to 0, so it's best to avoid squaring it, 
				# which is why i'll do some numerical gymnastics here (like Yarin Gal does)
				# sigma = tf.reshape(tf.exp(Ui), [-1,a])
				prec = tf.reshape(tf.exp(-ln_sigma2), [-1,a])
				sigma2 = 1.0/prec
				sigma = tf.sqrt(sigma2)
			with tf.variable_scope("mixing_coeffs", dtype=tf.float64):
				# regularize to keep activations before softmax low
				# otherwise they just keeps growing
				W = tf.get_variable('W', [15, 1], initializer=self._inits.uniform, regularizer=self._reg.l2(0.0001))
				b = tf.get_variable('b', [1], initializer=self._inits.zeros, regularizer=self._reg.l2(0.0001))
				# W = tf.get_variable('W', [15, 1], initializer=self._inits.uniform)
				# b = tf.get_variable('b', [1], initializer=self._inits.zeros)
				alpha_in = self.matmul(Oi, W) + b # (?,12,1)
				alpha = tf.reshape(tf.nn.softmax(alpha_in, dim=1), [-1,a])
			print(mu.shape, "mu", sigma.shape, "sigma", alpha.shape, "alpha")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("stdE", oe, tf.float64)
			
			# loss
			prE = tf.reduce_sum(mu*alpha, axis=-1)[:,None] # (?,1)
			print(prE.shape, "prE")
			prU = tf.reduce_sum(alpha*(sigma2 + (mu - prE)**2), axis=-1) # (?,), variance
			print(prU.shape, "prU")
 
			dy = (E-ue)/oe - mu # (?,12)
			print(dy.shape, "dy")
			phi = tf.exp(-( 0.5*prec*dy*dy ))/( sigma ) # (?,12)
			print(phi.shape, "phi") # (?,12)
			log_input = tf.reduce_sum(alpha*phi, axis=-1)
			# avoid log(0) by setting those points to log(1), which also makes their gradients 0
			log_input = tf.where(tf.equal(log_input, 0.0), tf.ones_like(log_input), log_input)
			logl = tf.log(log_input) # (?,)
			print(logl.shape, 'logl') # (?,)
			# negative_logl = tf.log((2*np.pi)**0.5 * o) + dE**2/(2 * o**2) # (?,1)
			
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = -tf.reduce_mean(logl) + reg_loss
			# loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			# momentum = 0.9
			# optimizer = tf.train.MomentumOptimizer(lr, momentum)
			optimizer = tf.train.AdamOptimizer(lr)
			train = optimizer.minimize(loss, global_step=step)
	
			# summary ops
			# tf.summary.scalar("loss", loss)
			# tf.summary.scalar("rmse", rmse)
			# tf.summary.scalar("mae", mae)
			# tf.summary.histogram("log_in", log_input)
			# tf.summary.histogram("alpha_in", alpha_in)
			# tf.summary.histogram("alpha", alpha)
			# tf.summary.histogram("mu", mu)
			# tf.summary.histogram("sigma", sigma)
			# tf.summary.histogram("C", C)
			# tf.summary.histogram("V", V)
			# tf.summary.histogram("Oi", Oi)
			# just in case i get NaN errors again
			# grad_alpha_in, grad_mu, grad_sigma = tf.gradients(loss, [alpha_in, mu, sigma])
			# tf.summary.histogram("grad_alpha_in", grad_alpha_in)
			# tf.summary.histogram("grad_mu", grad_mu)
			# tf.summary.histogram("grad_sigma", grad_sigma)

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


class ReDensityMixtureDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			step = tf.Variable(0, name='global_step', trainable=False)
			q = tf.shape(E)[0] # batch size (q, because it's printed as a question mark)
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
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				C = tf.gather(A, Z) # (?,12,30)
			with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f])
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
				Oi = tanh(self.matmul(C, W) + b)

			# predict means (energies), std (uncertainty) and priors (alpha)
			ys = []
			names = ["mean", "variance", "mixing_coeffs"]
			steps = [3,1,1]
			regs = [None, None, self._reg.l2(0.0001)]
			for scopename,T,reg in zip(names, steps, regs):
				with tf.variable_scope(scopename):
					with tf.variable_scope("out2", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b # (?,12,1)
						ys += [yi]
			mui, ln_sigma2i, alphai = ys
			mu = tf.reshape(mui, [-1,a])
			alpha = tf.reshape(tf.nn.softmax(alphai, dim=1), [-1,a])
			# sigma may come close to 0, so it's best to avoid squaring it, 
			# which is why i'll do some numerical gymnastics here (like Yarin Gal does)
			prec = tf.reshape(tf.exp(-ln_sigma2i), [-1,a])
			sigma2 = 1.0/prec
			sigma = tf.sqrt(sigma2)

			print(mu.shape, "mu", sigma.shape, "sigma", alpha.shape, "alpha")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("stdE", oe, tf.float64)
			
			# loss
			prE = tf.reduce_sum(mu*alpha, axis=-1)[:,None] # (?,1)
			print(prE.shape, "prE")
			prU = tf.reduce_sum(alpha*(sigma2 + (mu - prE)**2), axis=-1) # (?,), variance
			print(prU.shape, "prU")
 
			dy = (E-ue)/oe - mu # (?,12)
			print(dy.shape, "dy")
			phi = tf.exp(-( 0.5*prec*dy*dy ))/( sigma ) # (?,12)
			print(phi.shape, "phi") # (?,12)
			log_input = tf.reduce_sum(alpha*phi, axis=-1)
			# avoid log(0) by setting those points to log(1), which also makes their gradients 0
			log_input = tf.where(tf.equal(log_input, 0.0), tf.ones_like(log_input), log_input)
			logl = tf.log(log_input) # (?,)
			print(logl.shape, 'logl') # (?,)
			
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = tf.cond(
				epoch > 100,
				lambda: -tf.reduce_mean(logl) + reg_loss,
				lambda: tf.reduce_mean(dy**2),
				name="loss"
			)
			# loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
			# momentum = 0.9
			# optimizer = tf.train.MomentumOptimizer(lr, momentum)
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


class HalfDensityMixtureDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			step = tf.Variable(0, name='global_step', trainable=False)
			q = tf.shape(E)[0] # batch size (q, because it's printed as a question mark)
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
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				C = tf.gather(A, Z) # (?,12,30)
			with tf.variable_scope("interaction", initializer=self._inits.uniform, dtype=tf.float64):
				Wcf = tf.get_variable("Wcf", [c,f])
				b1 = tf.get_variable("b1", [f], initializer=self._inits.zeros)
				Wdf = tf.get_variable("Wdf", [d,f])
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

			# predict means (energies), std (uncertainty) and priors (alpha)
			ys = []
			names = ["mean", "variance", "mixing_coeffs"]
			steps = [3,1,1]
			regs = [None, None, self._reg.l2(0.0001)]
			for scopename,T,reg in zip(names, steps, regs):
				with tf.variable_scope(scopename):
					with tf.variable_scope("out1", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [c, 15])
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W) + b)
					with tf.variable_scope("out2", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b # (?,12,1)
						ys += [yi]
			mui, ln_sigma2i, alphai = ys
			mu = tf.reshape(mui, [-1,a])
			alpha = tf.reshape(tf.nn.softmax(alphai, dim=1), [-1,a])
			# sigma may come close to 0, so it's best to avoid squaring it, 
			# which is why i'll do some numerical gymnastics here (like Yarin Gal does)
			prec = tf.reshape(tf.exp(-ln_sigma2i), [-1,a])
			sigma2 = 1.0/prec
			sigma = tf.sqrt(sigma2)

			print(mu.shape, "mu", sigma.shape, "sigma", alpha.shape, "alpha")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("stdE", oe, tf.float64)
			
			# loss
			prE = tf.reduce_sum(mu*alpha, axis=-1)[:,None] # (?,1)
			print(prE.shape, "prE")
			prU = tf.reduce_sum(alpha*(sigma2 + (mu - prE)**2), axis=-1) # (?,), variance
			print(prU.shape, "prU")
 
			dy = (E-ue)/oe - mu # (?,12)
			print(dy.shape, "dy")
			phi = tf.exp(-( 0.5*prec*dy*dy ))/( sigma ) # (?,12)
			print(phi.shape, "phi") # (?,12)
			log_input = tf.reduce_sum(alpha*phi, axis=-1)
			# avoid log(0) by setting those points to log(1), which also makes their gradients 0
			log_input = tf.where(tf.equal(log_input, 0.0), tf.ones_like(log_input, dtype=tf.float64), log_input)
			logl = tf.log(log_input) # (?,)
			print(logl.shape, 'logl') # (?,)
			
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = tf.cond(
				epoch > 100,
				lambda: -tf.reduce_mean(logl) + reg_loss,
				lambda: tf.reduce_mean(((E-ue)/oe - prE)**2),
				name="loss"
			)
			# loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
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


class FullDensityMixtureDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			step = tf.Variable(0, name='global_step', trainable=False)
			q = tf.shape(E)[0] # batch size (q, because it's printed as a question mark)
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
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				AZ = tf.gather(A, Z) # (?,12,30)

			# predict means (energies), std (uncertainty) and priors (alpha)
			ys = []
			names = ["mean", "variance", "mixing_coeffs"]
			steps = [3,3,3]
			regs = [None, None, self._reg.l2(0.0001)]
			for scopename,T,reg in zip(names, steps, regs):
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

						DW = self.matmul(D, Wdf) + b2
						for t in range(T):
							CW = self.matmul(C, Wcf) + b1
							V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
							V = V*diag_zeros
							C += tf.reduce_sum(tanh(V), axis=2) # (qic)

					with tf.variable_scope("out1", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [c, 15])
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W) + b)
					with tf.variable_scope("out2", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b # (?,12,1)
						ys += [yi]
			mui, ln_sigma2i, alphai = ys
			mu = tf.reshape(mui, [-1,a])
			alpha = tf.reshape(tf.nn.softmax(alphai, dim=1), [-1,a])
			# sigma may come close to 0, so it's best to avoid squaring it, 
			# which is why i'll do some numerical gymnastics here (like Yarin Gal does)
			prec = tf.reshape(tf.exp(-ln_sigma2i), [-1,a])
			sigma2 = 1.0/prec
			sigma = tf.sqrt(sigma2)

			print(mu.shape, "mu", sigma.shape, "sigma", alpha.shape, "alpha")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("stdE", oe, tf.float64)
			
			# loss
			prE = tf.reduce_sum(mu*alpha, axis=-1)[:,None] # (?,1)
			print(prE.shape, "prE")
			prU = tf.reduce_sum(alpha*(sigma2 + (mu - prE)**2), axis=-1) # (?,), variance
			print(prU.shape, "prU")
 
			dy = (E-ue)/oe - mu # (?,12)
			print(dy.shape, "dy")
			phi = tf.exp(-( 0.5*prec*dy*dy ))/( sigma ) # (?,12)
			print(phi.shape, "phi") # (?,12)
			log_input = tf.reduce_sum(alpha*phi, axis=-1)
			# avoid log(0) by setting those points to log(1), which also makes their gradients 0
			log_input = tf.where(tf.equal(log_input, 0.0), tf.ones_like(log_input, dtype=tf.float64), log_input)
			logl = tf.log(log_input) # (?,)
			print(logl.shape, 'logl') # (?,)
			
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = tf.cond(
				epoch > 200,
				lambda: -tf.reduce_mean(logl) + reg_loss,
				lambda: tf.reduce_mean(((E-ue)/oe - prE)**2),
				name="loss"
			)
			# loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
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


class SteppedDensityMixtureDTNN(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, mu_max=10, tf_config=None, name=None):
		max_atoms = 20

		graph = tf.Graph()
		with graph.as_default():
			tanh = tf.nn.tanh
			# tanh = tf.nn.softplus

			R = tf.placeholder(tf.float64, shape=[None, None, 3], name="R") # input (atomic coordinates)
			Z = tf.placeholder(tf.int32, shape=[None, None], name="Z") # input (nuclear charges)
			E = tf.placeholder(tf.float64, shape=[None, 1], name="E") # output (energies per molecule)
			lr = tf.placeholder(tf.float64, name='learning_rate')
			keep = tf.placeholder(tf.float64, name='keep_probability')
			epoch = tf.placeholder(tf.int32, name='epoch')
			step = tf.Variable(0, name='global_step', trainable=False)
			q = tf.shape(E)[0] # batch size (q, because it's printed as a question mark)
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
			with tf.variable_scope("init", initializer=self._inits.normal, dtype=tf.float64):
				A = tf.get_variable("A", [max_atoms,c], initializer=self._inits.normal)
				AZ = tf.gather(A, Z) # (?,12,30)

			# predict means (energies), std (uncertainty) and priors (alpha)
			ys = []
			names = ["mean", "variance", "mixing_coeffs"]
			steps = [3,1,1]
			regs = [None, None, self._reg.l2(0.0001)]
			for scopename,T,reg in zip(names, steps, regs):
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

						DW = self.matmul(D, Wdf) + b2
						for t in range(T):
							CW = self.matmul(C, Wcf) + b1
							V = self.matmul(CW[:,None,:,:] * DW, Wfc) # (qijc)
							V = V*diag_zeros
							C += tf.reduce_sum(tanh(V), axis=2) # (qic)

					with tf.variable_scope("out1", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [c, 15])
						b = tf.get_variable("b", [15], initializer=self._inits.zeros)
						Oi = tanh(self.matmul(C, W) + b)
					with tf.variable_scope("out2", initializer=self._inits.uniform, regularizer=reg, dtype=tf.float64):
						W = tf.get_variable("W", [15,1])
						b = tf.get_variable("b", [1])
						yi = self.matmul(Oi, W) + b # (?,12,1)
						ys += [yi]
			mui, ln_sigma2i, alphai = ys
			mu = tf.reshape(mui, [-1,a])
			alpha = tf.reshape(tf.nn.softmax(alphai, dim=1), [-1,a])
			# sigma may come close to 0, so it's best to avoid squaring it, 
			# which is why i'll do some numerical gymnastics here (like Yarin Gal does)
			prec = tf.reshape(tf.exp(-ln_sigma2i), [-1,a])
			sigma2 = 1.0/prec
			sigma = tf.sqrt(sigma2)

			print(mu.shape, "mu", sigma.shape, "sigma", alpha.shape, "alpha")
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue, tf.float64)
				oe = self.const("stdE", oe, tf.float64)
			
			# loss
			prE = tf.reduce_sum(mu*alpha, axis=-1)[:,None] # (?,1)
			print(prE.shape, "prE")
			prU = tf.reduce_sum(alpha*(sigma2 + (mu - prE)**2), axis=-1) # (?,), variance
			print(prU.shape, "prU")
 
			dy = (E-ue)/oe - mu # (?,12)
			print(dy.shape, "dy")
			phi = tf.exp(-( 0.5*prec*dy*dy ))/( sigma ) # (?,12)
			print(phi.shape, "phi") # (?,12)
			log_input = tf.reduce_sum(alpha*phi, axis=-1)
			# avoid log(0) by setting those points to log(1), which also makes their gradients 0
			log_input = tf.where(tf.equal(log_input, 0.0), tf.ones_like(log_input, dtype=tf.float64), log_input)
			logl = tf.log(log_input) # (?,)
			print(logl.shape, 'logl') # (?,)
			
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = tf.cond(
				epoch > 100,
				lambda: -tf.reduce_mean(logl) + reg_loss,
				lambda: tf.reduce_mean(((E-ue)/oe - prE)**2),
				name="loss"
			)
			# loss = tf.reduce_mean(tf.square(dE))
			mse = tf.reduce_mean(tf.square((E-ue) - prE*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((E-ue) - prE*oe))

			# training
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
