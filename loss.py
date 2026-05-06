
import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):
	# y_true is pi_target
	# y_pred is logits
	
	pi_target = y_true
	logits = y_pred

	# Mask out invalid actions (though in trading all are usually valid)
	zero = tf.zeros(shape = tf.shape(pi_target), dtype=tf.float32)
	where = tf.equal(pi_target, zero)
	negatives = tf.fill(tf.shape(pi_target), -100.0) 
	logits = tf.where(where, negatives, logits)

	# 1. Cross Entropy Loss
	ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi_target, logits = logits)

	# 2. Entropy Regularization (Phase 4.1)
	probs = tf.nn.softmax(logits)
	entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
	
	entropy_coeff = 0.001
	loss = tf.reduce_mean(ce_loss - entropy_coeff * entropy)

	return loss
