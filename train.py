import tensorflow as tf
import numpy as np
from Action import Action
from lr import DelayedCosineDecay
import tqdm, random
import os, time, gc
from StochasticNet import *



physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def run_training(Lx, Ly, NUM_diffusion, kappa, lam, mb_size):
    Epoch = 20000
    shape = (Lx, Ly)
    path = 'model/phi4/kappa'+str(kappa)+'_L'+str(Lx)+'_'+str(NUM_diffusion)
    os.makedirs(path, exist_ok=True)

    V = np.prod(shape)

    initial_lr = 1e-3
    target_lr = 5e-5
    start_step = 5000
    decay_step = 10000

    lr = DelayedCosineDecay(initial_learning_rate=initial_lr,
                 decay_start_step=start_step,
                 decay_steps=decay_step,
                 target_learning_rate=target_lr,
                 alpha=None)


    stochasticnet = StochasticNet(Lx=Lx, Ly=Ly, use_symmetry=False)
    optimizer = tf.optimizers.Adam(learning_rate=lr,
                                    weight_decay=1e-2,
                                    beta_1=0.9,
                                    beta_2=0.99,
                                    )
        
    checkpoint = tf.train.Checkpoint(model=stochasticnet, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=Epoch)
#    checkpoint.restore(tf.train.latest_checkpoint(path)).expect_partial()

    action = lambda cfgs: Action(cfgs=cfgs, lam=0.022, k=kappa)


    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def _train_step():
        with tf.GradientTape() as tape:
            inputs_fin, logP, action_val = stochasticnet.ForwardDiffusion(mb_size, shape, NUM_diffusion, action)
            kl_matrix = logP + action_val
            kl, std = tf.nn.moments(kl_matrix, -1)
            magnet = tf.reduce_mean(tf.abs(inputs_fin))
            gradients = tape.gradient(kl, stochasticnet.trainable_variables)
        optimizer.apply_gradients(zip(gradients, stochasticnet.trainable_variables))
        return kl/V, std, magnet

    log_file_path = os.path.join(path, 'training_log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write("epoch\tkl\tstd\ttime(s)\n")

    Time = 0.
    tqdm_epoch = tqdm.trange(Epoch + 1)
    for epoch in tqdm_epoch:
        time1 = time.time()
        kl, std, magnet = _train_step()
        time2 = time.time()
        step_time = time2 - time1
        Time += step_time
        tqdm_epoch.set_description(f'KL:{kl:.5f}, std:{std:.5f}, magnet:{magnet:.5f}')
        if epoch % 250 == 0:
            manager.save()
            save_path = 'kappa'+str(kappa)+'L'+str(Lx)
            cfgs, logP, action_val = stochasticnet.ForwardDiffusion(4096, shape, NUM_diffusion, action)
            np.save(save_path + '/SPS.npy', cfgs)
            np.save(save_path+'/logp.npy', logP)
            np.save(save_path+'/action.npy', action_val)

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch}\t{kl.numpy():.6f}\t{std.numpy():.6f}\t{Time:.6f}\n")

        if np.isnan(kl.numpy()):
            print("NaN detected, stopping training")
            break

    del stochasticnet, optimizer, checkpoint, manager
    tf.keras.backend.clear_session()
    gc.collect()



if __name__ == '__main__':

    L_list = [64]
    Diffusion_list = [250]
    Kappa = [0.28]
    mb_size = 32
    Ly = 8
    for kappa in Kappa:
        for Lx in L_list:
            for N in Diffusion_list:
                print(f"Starting training for kappa={kappa}")
                try:
                    run_training(Lx, Ly, N, kappa, 0.022, mb_size)
                finally:
                    # 强制清理
                    tf.keras.backend.clear_session()
                 
                gc.collect()
                print(f"Finished training for kappa={kappa}")
