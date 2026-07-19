from DiffusionSubNet import *
import tensorflow_probability as tfp
import tensorflow as tf
from Action import *
import numpy as np
import tqdm, random
from Symmerty import *
from Time_embedding import embedding 
tfd = tfp.distributions
tfl = tf.keras.layers


class StochasticNet(tf.keras.Model):
        
    def __init__(self, embed_dim=128, Lx=64, Ly=8, use_symmetry=False):
        super().__init__()
        self.time_embedding = embedding(embed_dim, tf.nn.relu)
        self.forward_net = DiffusionSubNet(self.time_embedding, Lx, Ly)
        self.backward_net = DiffusionSubNet(self.time_embedding, Lx, Ly)
        self.sum = lambda x: tf.reduce_sum(x, axis=tf.range(1, tf.rank(x)))
        self.diffusion_NN = NeuralDiffusionSchedule(self.time_embedding)
        self.symmetry = use_symmetry

    def forward(self, inputs_curr, time, num_diffusion):
        """
            Backward transition (reverse process).
    
            Parameters
            ----------
                inputs_curr : tf.Tensor, shape [batch, Lx, Ly]
                            Current state.
                time : tf.Tensor, shape [batch]
                     Normalized diffusion time.
                dt : float
                    Time step.

            Returns
            -------
                inputs_pred : tf.Tensor, shape [batch, Lx, Ly]
                            Predicted previous state (with symmetry applied).
                logp : tf.Tensor, shape [batch]
                        Log-probability ratio log p_fwd - log p_bwd.
        """
        esp =  tf.cast(1 / num_diffusion, dtype=inputs_curr.dtype)
        time = tf.cast(tf.fill([tf.shape(inputs_curr)[0]], time/num_diffusion), 
                                                            dtype=inputs_curr.dtype)

        diffusion = self.diffusion_NN(time) * tf.sqrt(esp)

#        inputs_curr = tf.expand_dims(inputs_curr, -1)
        drift_forw = self.forward_net(inputs_curr, time)
        drift_forw = drift_forw * diffusion**2
        p_forw = tfd.Normal(inputs_curr + drift_forw, diffusion)
        inputs_next = p_forw.sample()

        drift_back = self.backward_net(inputs_next, time)
        drift_back = drift_back * diffusion**2 
        p_back = tfd.Normal(inputs_next + drift_back, diffusion)

        logp_forw = p_forw.log_prob(inputs_next) 
        logp_back = p_back.log_prob(inputs_curr)
        if self.symmetry:
            inputs_next = symmetry(inputs_next)
#        inputs_next = tf.squeeze(inputs_next, -1)

        return inputs_next, self.sum(logp_forw - logp_back)

    def backward(self, inputs_curr, time, num_diffusion):
        """
            Backward transition (reverse process).
    
            Parameters
            ----------
                inputs_curr : tf.Tensor, shape [batch, Lx, Ly]
                    Current state.
                time : tf.Tensor, shape [batch]
                    Normalized diffusion time.
                dt : float
                    Time step.

            Returns
            -------
                inputs_pred : tf.Tensor, shape [batch, Lx, Ly]
                             Predicted previous state (with symmetry applied).
                logp : tf.Tensor, shape [batch]
                        Log-probability ratio log p_fwd - log p_bwd.
        """


        esp =  tf.cast(1 / num_diffusion, dtype=inputs_curr.dtype)
        time = tf.cast(tf.fill([tf.shape(inputs_curr)[0]], time/num_diffusion),
                                                            dtype=inputs_curr.dtype)

        diffusion = self.diffusion_NN(time) * tf.sqrt(esp)

#        inputs_curr = tf.expand_dims(inputs_curr, -1)
        drift_back = self.backward_net(inputs_curr, time)
        drift_back = drift_back * diffusion**2
        p_back = tfd.Normal(loc=inputs_curr + drift_back, scale=diffusion)

        inputs_pred = p_back.sample()

        drift_forw = self.forward_net(inputs_pred, time)
        drift_forw = drift_forw * diffusion**2
        p_forw = tfd.Normal(loc=inputs_pred + drift_forw, scale=diffusion)

        logp_forw = p_forw.log_prob(inputs_curr)
        logp_back = p_back.log_prob(inputs_pred)
        if self.symmetry:
            inputs_pred = symmetry(inputs_pred)
 #       inputs_pred = tf.squeeze(inputs_pred, -1)
        
        return inputs_pred, self.sum(logp_forw - logp_back)

    def ForwardDiffusion(self, mb_size, shape, num_diffusion, action_fn):
        """
            Perform full forward diffusion process with KL objective.

            Parameters
            ----------
                mb_size : int
                        Mini-batch size.
                shape : tuple of int, e.g. (Lx, Ly)
                        Lattice shape per sample.
                num_diffusion : int
                        Number of diffusion steps.

            Returns
            -------
                inputs_curr : tf.Tensor, shape [mb_size, Lx, Ly]
                            Final state after diffusion.
                kl : tf.Tensor, shape [mb_size]
                    KL loss term combining forward log-probabilities, action, and initial prior.
        """
        base_dist = tfd.Normal(loc=tf.zeros(shape), scale=tf.ones(shape))
        base_dist = tfd.Independent(base_dist, reinterpreted_batch_ndims=2)
        inputs_curr = base_dist.sample(mb_size)
        logP = base_dist.log_prob(inputs_curr)
        
        self.Lx, self.Ly = shape[0], shape[1]
        self.mb_size = mb_size

        inputs_curr = tf.expand_dims(inputs_curr, -1)

        for time in tf.range(1, num_diffusion+1):
            inputs_curr, logp = self.forward(inputs_curr, time, num_diffusion)
            logP += logp

        inputs_curr = tf.squeeze(inputs_curr, -1)
        action = self.sum(action_fn(inputs_curr))

        return inputs_curr, logP, action



