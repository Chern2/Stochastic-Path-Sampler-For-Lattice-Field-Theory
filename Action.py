import tensorflow as tf


def Action(cfgs, lam=0.022, k=0.27):
    k = tf.cast(k, cfgs.dtype)
    lam = tf.cast(lam, cfgs.dtype)
    action = (1-2*lam)*tf.math.pow(cfgs,2) + lam*tf.math.pow(cfgs,4)
    for mu in range(1, len(cfgs.shape)):
        action -= 2*k*cfgs*tf.roll(cfgs, 1, axis=mu)
    return action



