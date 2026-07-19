import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Layer

tfl = tf.keras.layers

class embedding(tf.keras.Model):
    def __init__(self, embed_dim, activation=None):
        super().__init__()
        self.W = self.add_weight(
            name="W",
            shape=(embed_dim,),
            initializer=tf.random_normal_initializer(stddev=1.0),
            trainable=False
        )

        self.dense =  tfl.Dense(embed_dim//2, activation=activation,)

    def embedding(self, timesteps):
        pi = tf.cast(2*np.pi * self.W, dtype=timesteps.dtype)
        times = tf.expand_dims(timesteps, -1) * pi
        times = tf.concat([tf.sin(times), tf.cos(times)], axis=-1)
        return self.dense(times)

