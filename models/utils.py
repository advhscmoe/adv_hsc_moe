import tensorflow as tf

def get_capped_min_max_norm(x, fmin, fmax):
    return (tf.maximum(fmin, tf.minimum(tf.cast(x, tf.float32), fmax)) -
            fmin) / (fmax - fmin)

def log_x_plus_one(input_tensor):
    y = tf.cast(input_tensor, tf.float32)
    return tf.sign(y) * tf.log(tf.abs(y) + 1.0)

def log_min_max(x, fmin, fmax):
    if isinstance(x, tf.SparseTensor):
        x_value = x.values
    else:
        x_value = x
    x_log = log_x_plus_one(x_value)
    fmin_log = log_x_plus_one(fmin)
    fmax_log = log_x_plus_one(fmax)
    x_value = get_capped_min_max_norm(x_log, fmin_log, fmax_log)
    if isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(
            indices=x.indices, values=x_value, dense_shape=x.dense_shape)
    else:
        return x_value