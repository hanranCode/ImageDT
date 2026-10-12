import math

from imagedt._backends import tensorflow_v1

tf = tensorflow_v1()


def smooth_l1_loss(y_true, y_pred):
    delta = 0.5
    x = tf.abs(y_true - y_pred)
    loss = tf.where(x < delta, 0.5 * x**2, delta * (x - 0.5 * delta))
    return loss


def arcface_loss(embedding, labels, out_num, w_init=None, s=64.0, m=0.5):
    """
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    """
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope("arcface_loss"):
        # inputs and weights norm
        embedding = tf.nn.l2_normalize(embedding, axis=1, name="norm_embedding")
        weights = tf.get_variable(
            name="embedding_weights",
            shape=(embedding.get_shape().as_list()[-1], out_num),
            initializer=w_init,
            dtype=tf.float32,
        )
        weights = tf.nn.l2_normalize(weights, axis=0, name="norm_weights")
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name="cos_t")
        cos_t2 = tf.square(cos_t, name="cos_2")
        sin_t2 = tf.maximum(1.0 - cos_t2, 0.0, name="sin_2")
        sin_t = tf.sqrt(sin_t2, name="sin_t")
        cos_mt = s * tf.subtract(
            tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name="cos_mt"
        )

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name="if_else"), dtype=tf.bool)

        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(tf.reshape(labels, [-1]), depth=out_num, name="one_hot_mask")
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1.0, mask, name="inverse_mask")

        s_cos_t = tf.multiply(s, cos_t, name="scalar_cos_t")

        output = tf.add(
            tf.multiply(s_cos_t, inv_mask),
            tf.multiply(cos_mt_temp, mask),
            name="arcface_loss_output",
        )
    return output
