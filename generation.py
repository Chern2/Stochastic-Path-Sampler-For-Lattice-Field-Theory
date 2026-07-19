from StochasticNet import StochasticNet
from Action import Action
import tensorflow as tf
import numpy as np
from tqdm.auto import trange
import tensorflow_probability as tfp
import numpy as np
import os, time
import os


NUM_diffusion = 250
       
kappa = 0.28

lam = 0.022
mb_size = 1024 * 4
L = 64

Step_generation =  20*NUM_diffusion

N = 1
path = 'model/phi4/kappa'+str(kappa)+'_L'+str(L)+'_'+str(NUM_diffusion)
shape = (L, 8)

V = np.prod(shape)

latest = tf.train.latest_checkpoint(path)
print('xxxxxxxxxxxxxxxx', latest, 'xxxxxxxxxx')
action = lambda cfgs: Action(cfgs, lam=lam, k=kappa)
stochasticnet = StochasticNet(Lx=L, Ly=8)
checkpoint = tf.train.Checkpoint(model=stochasticnet)
checkpoint.restore(latest).expect_partial()

save_path = 'kappa'+str(kappa)+'L'+str(L)
os.makedirs(save_path, exist_ok=True)
cfgs, logP, action_val = stochasticnet.ForwardDiffusion(mb_size, shape, Step_generation, action)
np.save(save_path + '/DM.npy', cfgs)
np.save(save_path+'/logp.npy', logP)
np.save(save_path+'/action.npy', action_val)
print(f"Generated {len(cfgs)} configurations saved to cfgs.npy and kl.npy")
