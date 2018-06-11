import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class SimpleMixtureNetwork(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, tf_config=None, name=None):
		a = point['R'].shape[1]

		graph = tf.Graph()
		with graph.as_default():
			X = tf.placeholder(tf.float32, shape=[None, a, 3], name="X") # input
			Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y") # labels
			lr = tf.placeholder(tf.float32, name='learning_rate')
			keep = tf.placeholder(tf.float32, name='keep_probability')
			is_training = tf.placeholder(tf.bool, name='is_training')
			step = tf.Variable(0, name='global_step', trainable=False)

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: kw.get('keep', 1.0),
					is_training: kw.get('is_training', False),
					X: sample["R"].reshape([-1,a,3]),
					Y: sample["E"][:, None]
				}

			# forward feed
			identity = lambda x: x
			layers = [
				(a*a, identity, self._reg.l2(0.001)),
				(400, tf.nn.sigmoid, self._reg.l2(0.001)),
			]
			with tf.variable_scope("preprocessing"):
				D = self._interatomic_dists(X)
				y = tf.contrib.layers.batch_norm(
					tf.reshape(D, [-1, a*a]), is_training=is_training,
					center=True, scale=True, epsilon=1e-3, decay=0.99, renorm=True
				)
				y = y*tf.cast(y <= 20, tf.float32)
			for i, ((n1, _, _), (n2, f, r)) in enumerate(zip(layers[:-1], layers[1:])):
				with tf.variable_scope("layer%i"%i, regularizer=r):
					W = tf.get_variable("W", [n1, n2], initializer=self._inits.normal)
					b = tf.get_variable("b", [n2], initializer=self._inits.zeros)
					y = f(tf.matmul(y, W) + b)

			# conditional density layers, see bishop1995 ch. 6.4
			m = 12 # n_components
			c = 1 # dims(Y)
			with tf.variable_scope("mixing_coeffs", regularizer=self._reg.l2(0.001)):
				W = tf.get_variable('W', [n2, m], initializer=self._inits.uniform)
				b = tf.get_variable('b', [m], initializer=self._inits.zeros)
				alpha = tf.nn.softmax(tf.matmul(y, W) + b) # (?,m)
			with tf.variable_scope("scales", regularizer=self._reg.l2(0.001)):
				W = tf.get_variable('W', [n2, m], initializer=self._inits.uniform)
				b = tf.get_variable('b', [m], initializer=self._inits.zeros)
				sigma = tf.exp(tf.matmul(y, W) + b) # (?,m)
			with tf.variable_scope("centers", regularizer=self._reg.l2(0.001)):
				W = tf.get_variable('W', [n2, m], initializer=self._inits.uniform)
				b = tf.get_variable('b', [m], initializer=self._inits.zeros)
				mu = tf.matmul(y, W) + b # (?,m)
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const('meanE', ue)
				oe = self.const('stdE', oe)

			# loss
			# print(alpha.shape, mu.shape, sigma.shape, "alpha mu sigma")
			prY = tf.reduce_sum(mu*alpha, axis=-1)[:, None]
			print((mu - prY).shape, "mu-prY", sigma.shape, 'sigma', alpha.shape, 'alpha')
			prU = tf.reduce_sum(alpha*(sigma**2 + (mu - prY)**2), axis=-1)[:, None]
			print(prU.shape, "prU")

			dy = (Y-ue)/oe - mu
			phi = tf.exp(-( dy**2 )/( 2*sigma**2 )) / ( (2*np.pi)**(c/2) * sigma**c )
			# logl = tf.log(tf.reduce_sum(alpha*phi, axis=-1))

			log_input = tf.reduce_sum(alpha*phi, axis=-1)
			# avoid log(0) by setting those points to log(1), which also makes their gradients 0
			log_input = tf.where(tf.equal(log_input, 0.0), tf.ones_like(log_input), log_input)
			logl = tf.log(log_input) # (?,)

			# reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			# self._loss = tf.reduce_mean(tf.square((Y-ue) - prY*oe)) + reg_loss
			loss = -tf.reduce_mean(logl)
			mse = tf.reduce_mean(tf.square((Y-ue) - prY*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((Y-ue) - prY*oe))

			# training
			batch_norm_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(batch_norm_update): # attach running-avg-update to train op
				# momentum = 0.98
				# optimizer = tf.train.MomentumOptimizer(lr, momentum)
				# optimizer = tf.train.GradientDescentOptimizer(lr)
				optimizer = tf.train.AdamOptimizer(lr)
				train = optimizer.minimize(loss, global_step=step)

			# summary ops
			# tf.summary.scalar("loss", self._loss)
			# tf.summary.scalar("rmse", self._rmse)
			# tf.summary.scalar("mae", self._mae)

			super().__init__(
				tf_config = tf_config,
				name = name,
				graph = graph,
				train_op = train,
				fill_feed = fill_feed,
				losses = dict(loss=loss, rmse=rmse, mae=mae),
				targets = dict(E=prY*oe + ue, U=tf.sqrt(prU)*oe),
				targets_raw = dict(E=prY, U=tf.sqrt(prU)),
			)

