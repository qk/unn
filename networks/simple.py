import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

class SimpleNetwork(NeuralNet, MolecularPreprocessingMixin):
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
					X: sample["R"].reshape([-1, a, 3]),
					Y: sample["E"][:,None]
				}

			# forward feed
			identity = lambda x: x
			layers = [
				(a*a, identity, self._reg.l2(0.001)),
				(400, tf.nn.sigmoid, self._reg.l2(0.001)),
				(1, identity, None)
			]
			with tf.variable_scope("preprocessing"):
				D = self._interatomic_dists(X)
				# batch_norm() normalizes over all except the last axis, so put every axis' that shouldn't be normalized over there. also note the required extra dependencies handling near the train-op creation, otherwise validation error will be high, because the exponential averaging won't work.
				y = tf.contrib.layers.batch_norm(
					tf.reshape(D, [-1, a*a]), is_training=is_training,
					center=True, scale=True, epsilon=1e-3, decay=0.99, renorm=True
				)
				y = y*tf.cast(y <= 20, tf.float32)
				# y = y*tf.cast(y >= 1e-6, tf.float32)
			for i,((n1, _, _), (n2, f, r)) in enumerate(zip(layers[:-1], layers[1:])):
				with tf.variable_scope("layer%i"%i, regularizer=r):
					W = tf.get_variable("W", [n1, n2], initializer=self._inits.normal)
					b = tf.get_variable("b", [n2], initializer=self._inits.zeros)
					y = f(tf.matmul(y, W) + b)
			prY = y
			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue)
				oe = self.const("stdE", oe)

			# loss
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = tf.reduce_mean(tf.square(prY*oe - (Y-ue))) + 10*reg_loss
			mse = tf.reduce_mean(tf.square(prY*oe - (Y-ue)))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs(prY*oe - (Y-ue)))

			# training
			batch_norm_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(batch_norm_update): # attach running-avg-update to train op
				# lr = tf.train.exponential_decay(lr, step, 200, 0.96)
				momentum = 0.99
				optimizer = tf.train.MomentumOptimizer(lr, momentum)
				# optimizer = tf.train.AdamOptimizer(lr)
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
				targets_raw = dict(E=prY),
				targets = dict(E=prY*oe + ue)
			)

# train to <0.08 rmse:
# 50k tr points, 5k va points, all randomly chosen (but disjoint) from the benzene testset, then
# .fit(tr, va, 200, 1000, lr=exp_lr(1e-3, 1e-5, 1000, floor=5e-5), silent=False)
