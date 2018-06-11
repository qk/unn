import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class SimpleDensityNetwork(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, tf_config=None, name=None):
		a = point["R"].shape[1]

		graph = tf.Graph()
		with graph.as_default():
			X = tf.placeholder(tf.float32, shape=[None, a, 3], name="X") # input
			Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y") # labels
			lr = tf.placeholder(tf.float32, name='learning_rate')
			is_training = tf.placeholder(tf.bool, name='is_training')

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					# keep: kw.get('keep', 1.0),
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
			for i,((n1, _, _), (n2, f, r)) in enumerate(zip(layers[:-1], layers[1:])):
				with tf.variable_scope("layer%i"%i, regularizer=r):
					W = tf.get_variable("W", [n1, n2], initializer=self._inits.normal)
					b = tf.get_variable("b", [n2], initializer=self._inits.zeros)
					y = f(tf.matmul(y, W) + b)

			# conditional density layers, see bishop1995 ch. 6.4
			with tf.variable_scope("scales", regularizer=self._reg.l2(0.001)):
				W = tf.get_variable('W', [n2, 1], initializer=self._inits.uniform)
				b = tf.get_variable('b', [1], initializer=self._inits.zeros)
				ls2 = tf.matmul(y, W) + b # log sigma**2
				prec = tf.exp(-ls2) # (?,1)
			with tf.variable_scope("centers", regularizer=self._reg.l2(0.001)):
				W = tf.get_variable('W', [n2, 1], initializer=self._inits.uniform)
				b = tf.get_variable('b', [1], initializer=self._inits.zeros)
				mu = tf.matmul(y, W) + b # (?,1)
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("mu_E", ue)
				oe = self.const("sigma_E", oe)

			# loss
			prY = mu
			prU = 1./prec # sigma**2

			dy = (Y-ue) - mu*oe # (?,1)
			# phi = tf.exp(-( dy**2 )/( 2*sigma**2 )) / ( (2*np.pi)**(c/2) * sigma**c )
			# print(phi.shape, "phi") # (?,12)
			# logl = tf.log(tf.reduce_sum(alpha*phi, axis=-1))
			# negative_logl = tf.log((2*np.pi)**0.5 * sigma) + dy**2/(2*sigma**2) # (?,1)
			# negative_logl = tf.log(sigma) + dy**2/(2*sigma**2) # (?,1)
			# negative_logl = ls + dy**2/(2*sigma**2) # (?,1)
			negative_logl = ls2 + prec*dy*dy # (?,1)

			# reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			# self._loss = tf.reduce_mean(tf.square((Y-ue) - prY*oe)) + reg_loss
			loss = tf.reduce_mean(negative_logl)
			mse = tf.reduce_mean(tf.square((Y-ue) - prY*oe))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs((Y-ue) - prY*oe))

			# training
			batch_norm_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(batch_norm_update): # attach running-avg-update to train op
				step = tf.Variable(0, name='global_step', trainable=False)
				lr = tf.train.exponential_decay(lr, step, 400, 0.96)
				momentum = 0.99
				optimizer = tf.train.MomentumOptimizer(lr, momentum)
				# optimizer = tf.train.GradientDescentOptimizer(lr)
				# optimizer = tf.train.AdamOptimizer(lr, momentum)
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
				targets = dict(E=prY*oe + ue, U=tf.sqrt(prU)*oe),
				targets_raw = dict(E=prY, U=tf.sqrt(prU)),
			)
