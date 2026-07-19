import tensorflow as tf
import numpy as np
from Action import *
from CyclicConv2D import CyclicConv2D



tfl = tf.keras.layers


class DiffusionSubNet(tf.keras.Model):
    """
    Diffusion drift network for an L x 8 periodic lattice.

    Parameters
    ----------
    embed
        Object providing embed.embedding(time).
    L : int
        Lattice extent in the first spatial direction.
    channels : int
        Number of hidden feature channels.
    debug : bool
        Print layer/kernel shapes during the first build.
    """

    def __init__(self, embed, Lx, Ly, channels=32):
        super().__init__()

        self.embed = embed
        self.channels = int(channels)

        L_y = Ly // 4 + 1
        L_x = Lx // 4 + 1

        self.conv1 = CyclicConv2D(
            filters=channels,
            kernel_size=(L_x, L_y),
            strides=(1, 1),
            use_bias=False,
            activation=None,
            name="conv1",)

        self.conv2 = CyclicConv2D(
            filters=channels,
            kernel_size=(L_x, L_y),
            strides=(1, 1),
            use_bias=False,
            activation=None,
            name="conv2",)

        self.conv3 = CyclicConv2D(
            filters=channels,
            kernel_size=(L_x, L_y),
            strides=(1, 1),
            use_bias=False,
            activation=None,
            name="conv3",)


        self.drift = CyclicConv2D(
            filters=1,
            kernel_size=(L_x, L_y),
            strides=(1, 1),
            use_bias=False,
            activation=None,
            name="drift",)


        # Time-conditioning projections
        self.dense1 = tfl.Dense(self.channels, activation='tanh')
        self.dense2 = tfl.Dense(self.channels, activation='tanh')
        self.dense3 = tfl.Dense(self.channels, activation='tanh')


        self.Batch1 = tfl.BatchNormalization(center=False)
        self.Batch2 = tfl.BatchNormalization(center=False)
        self.Batch3 = tfl.BatchNormalization(center=False)

        self.act = lambda x: tf.clip_by_value(x, -1.0, 1.0)


    @staticmethod
    def _broadcast_time_embedding(time_embedding):
        """
        Convert [batch, channels] to [batch, 1, 1, channels].
        """
        return time_embedding[:, tf.newaxis, tf.newaxis, :]


    def network(self, inputs, time, use_d4=True):
        """
        Evaluate the drift field.

        Parameters
        ----------

        inputs : tf.Tensor
            Shape [batch, L, 8, channels_in].
        time : tf.Tensor
            Scalar or shape [batch].

        Returns
        -------
        tf.Tensor
            Drift field with shape [batch, L, 8, 1].
        """
        time_embedding = self.embed.embedding(time)

        time_embedding1 = self._broadcast_time_embedding(
            self.dense1(time_embedding)
        )
        time_embedding2 = self._broadcast_time_embedding(
            self.dense2(time_embedding)
        )
        time_embedding3 = self._broadcast_time_embedding(
            self.dense3(time_embedding)
        )

        h1 = self.conv1(inputs, use_d4=use_d4)
        h1 = self.Batch1(h1)
        h1 = h1 * tf.exp(time_embedding1)
        h1 = self.act(h1) 

        h2 = self.conv2(h1, use_d4=use_d4)
        h2 = self.Batch2(h2)
        h2 = h2 * tf.exp(time_embedding2)
        h2 = self.act(h2) + h1

        h3 = self.conv3(h2, use_d4=use_d4)
        h3 = self.Batch3(h3)
        h3 = h3 * tf.exp(time_embedding3)
        h3 = self.act(h3) + h2

        drift = self.drift(h3, use_d4=use_d4)

        return drift

    def call(self, inputs, time):
        drift = self.network(inputs, time)
        return drift/2

class NeuralDiffusionSchedule(tf.keras.layers.Layer):
    """
    Learnable time-dependent diffusion schedule.

    The final layer is zero-initialized, so initially r_theta(t)=0
    and the schedule is exactly the original hand-written one.
    """

    def __init__(
        self,
        embed,
        hidden_dim = 64,
        name = "neural_diffusion_schedule",):

        super().__init__(name=name)
        self.embed = embed
        self.hidden_dim = int(hidden_dim)
        self.dense = tfl.Dense(hidden_dim, activation='relu',)
        self.out = tfl.Dense(1, activation='sigmoid')

    def call(self, time):
        time_embd = self.embed.embedding(time)
        h = self.dense(time_embd)
        mult = self.out(h)
        mult = mult[:, None, None, :]
        return  mult


