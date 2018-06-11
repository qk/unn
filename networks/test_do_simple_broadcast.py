import numpy as np
import tensorflow as tf
from networks.neuralnet import NeuralNet, MolecularPreprocessingMixin

# test if broadcasting is significantly faster

def q_rows(W, keep=0.9, n=100):
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

class SimpleHomoscedastic(NeuralNet, MolecularPreprocessingMixin):
	def __init__(self, tr_mu_sigma, point, trN, keep_prob=0.95, length_scale=1, n_predictions=10, tf_config=None, name=None):
		a = point['R'].shape[1]

		graph = tf.Graph()
		with graph.as_default():
			X = tf.placeholder(tf.float32, shape=[None, a, 3], name="X") # input
			Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y") # labels
			lr = tf.placeholder(tf.float32, name='learning_rate')
			keep = tf.placeholder(tf.float32, name='keep_probability')
			is_training = tf.placeholder(tf.bool, name='is_training')
			step = tf.Variable(0, name='global_step', trainable=False)
			weight_decay = self.const("weight_decay", 1e-4)
			n_points = self.const("n_training_points", trN)
			scale2 = self.const("length_scale_squared", length_scale**2)
			__t = 2*n_points*weight_decay/keep_prob/scale2  # inverse precision, 1/tau

			def fill_feed(sample, **kw):
				return {
					lr: kw.get('lr', 1e-3),
					keep: keep_prob, # always drop
					is_training: kw.get('is_training', False),
					X: sample["R"].reshape([-1, a, 3]),
					Y: sample["E"][:,None],
				}

			with tf.variable_scope("preprocessing"):
				D = self._interatomic_dists(X)
				D = tf.contrib.layers.batch_norm(
					tf.reshape(D, [-1, a*a]), is_training=is_training,
					center=True, scale=True, epsilon=1e-3, decay=0.99, renorm=True
				)
				D = D*tf.cast(D <= 20, tf.float32)

			# forward feed n_predictions times (matrix broadcast)
			y = tf.tile(D[None,...], (n_predictions,1,1))
			print(y.shape, 'y in')
			identity = lambda x: x
			layers = [
				(a*a, identity, self._reg.l2(weight_decay)),
				(int(400/keep_prob), tf.nn.sigmoid, self._reg.l2(weight_decay)),
				(1, identity, None)
			]
			for i,((n1, _, _), (n2, f, r)) in enumerate(zip(layers[:-1], layers[1:])):
				with tf.variable_scope("layer%i"%i, regularizer=r, reuse=tf.AUTO_REUSE):
					W = q_rows(
						tf.get_variable("W", [n1, n2], initializer=self._inits.normal),
						keep = 1.0 if i == 0 else keep,
						n = n_predictions
					)
					b = tf.get_variable("b", [n2], initializer=self._inits.zeros)
					print("y", y.shape, "dot W ", W.shape, "+ b", b.shape)
					y = f(tf.matmul(y, W) + b)

			with tf.variable_scope("scaling"):
				ue, oe = tr_mu_sigma["E"]
				ue = self.const("meanE", ue)
				oe = self.const("stdE", oe)

			print(y.shape, "result")
			prY, prV = tf.nn.moments(y, axes=[0]) # mean, variance
			trueY = (Y - ue)/oe

			# loss
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			loss = tf.reduce_mean(0.5*tf.square(trueY - prY)) + 2*reg_loss # reg_loss has extra 0.5 factor by default, so actual weight decay would've been 0.5*weight_decay if not for the extra factor 2
			mse = tf.reduce_mean(tf.square(prY*oe - (Y-ue)))
			rmse = tf.sqrt(mse)
			mae = tf.reduce_mean(tf.abs(prY*oe - (Y-ue)))

			# training
			batch_norm_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(batch_norm_update): # attach running-avg-update to train op
				momentum = 0.99
				optimizer = tf.train.MomentumOptimizer(lr, momentum)
				# optimizer = tf.train.GradientDescentOptimizer(lr)
				# optimizer = tf.train.AdamOptimizer(lr, beta1=0.99, epsilon=0.01)
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
				targets = dict(E=prY*oe + ue, U=(prV + __t)*oe**2), # *oe**2 probably unnecessary
				targets_raw = dict(E=prY, U=prV),
			)
