# Stochastic Path Sampler for Lattice Field Theory

Official TensorFlow implementation of the **Stochastic Path Sampler (SPS)** for lattice field theory.

This repository contains the code associated with:

> **Stochastic Path Sampler for Lattice Field Theory**  
> Shiyang Chen, Moxian Qian, Gert Aarts, Biagio Lucini, and Kai Zhou  
> arXiv:2606.13790 [hep-lat]

- arXiv: <https://arxiv.org/abs/2606.13790>
- DOI: <https://doi.org/10.48550/arXiv.2606.13790>

## Overview

The Stochastic Path Sampler is a generative sampler for lattice field theory based on learnable nonequilibrium stochastic dynamics.

SPS constructs a forward stochastic process and an auxiliary backward stochastic process connecting a tractable prior distribution to an unnormalized target distribution,

$$
\pi(\phi)
=
\frac{1}{Z}e^{-S(\phi)}.
$$

The model is trained by minimizing a path-space variational objective defined through the log-ratio of the forward and backward trajectory measures. After training, the learned forward process can generate proposals for the target lattice field distribution.

This implementation applies SPS to the two-dimensional scalar $\phi^4$ theory with periodic boundary conditions.

## Lattice action

The lattice action implemented in this repository is

$$
S[\phi]
=
\sum_x
\left[
(1-2\lambda)\phi_x^2
+
\lambda\phi_x^4
-
2\kappa
\sum_{\mu}
\phi_x\phi_{x+\hat{\mu}}
\right].
$$

Here:

- $\kappa$ is the hopping parameter;
- $\lambda$ is the quartic coupling;
- periodic boundary conditions are imposed in every lattice direction.

The action is implemented in `Action.py`.

## Stochastic dynamics

For a discretization depth $N$, the forward transition is represented schematically by

$$
\phi_{i+1}
=
\phi_i
+
\sigma_\theta^2(t_i)
K_{\theta,\mathrm{F}}(\phi_i,t_i)
+
\sigma_\theta(t_i)\sqrt{\Delta t}\,\eta_i,
$$

where

$$
\eta_i\sim\mathcal{N}(0,I),
\qquad
\Delta t=\frac{1}{N}.
$$

An auxiliary backward transition is defined using a second drift network,

$$
\phi_i
=
\phi_{i+1}
+
\sigma_\theta^2(t_i)
K_{\theta,\mathrm{B}}(\phi_{i+1},t_i)
+
\sigma_\theta(t_i)\sqrt{\Delta t}\,\widetilde{\eta}_i.
$$

The trajectory-dependent endpoint log-density estimator is accumulated as

$$
\log q_{\mathrm{SPS}}(\phi_N;\tau)
=
\log \pi_0(\phi_0)
+
\sum_{i=0}^{N-1}
\left[
\log q_{\mathrm{F}}(\phi_{i+1}\mid\phi_i)
-
\log q_{\mathrm{B}}(\phi_i\mid\phi_{i+1})
\right].
$$

The training objective is the reverse path-space KL estimator

$$
\mathcal{L}_{\mathrm{SPS}}
=
\mathbb{E}_{q_{\mathrm{F}}}
\left[
\log q_{\mathrm{SPS}}(\phi_N;\tau)
+
S(\phi_N)
\right],
$$

up to the unknown normalization constant of the target distribution.

## Repository structure

```text
.
├── Action.py
├── CyclicConv2D.py
├── DiffusionSubNet.py
├── StochasticNet.py
├── Symmerty.py
├── Time_embedding.py
├── lr.py
├── train.py
├── generation.py
└── README.md
```

## Code description

### `Action.py`

Implements the two-dimensional scalar $\phi^4$ lattice action with periodic nearest-neighbour interactions.

### `CyclicConv2D.py`

Implements two-dimensional convolution with periodic padding.

The layer supports:

- periodic boundary conditions;
- square and rectangular kernels;
- optional $D_4$ averaging for square kernels;
- optional $D_2$ averaging for rectangular kernels;
- grouped convolution.

### `DiffusionSubNet.py`

Defines the neural drift network used in the forward and backward stochastic processes.

The architecture contains:

- periodic convolutional layers;
- residual connections;
- Fourier time embeddings;
- multiplicative time conditioning;
- batch normalization;
- bounded nonlinear activations;
- a learnable time-dependent diffusion schedule.

### `StochasticNet.py`

Implements the complete SPS model, including:

- Gaussian prior sampling;
- forward stochastic transitions;
- auxiliary backward transitions;
- trajectory log-ratio accumulation;
- terminal-action evaluation;
- full forward trajectory generation.

### `Symmerty.py`

Implements random lattice symmetry transformations, including:

- global $\mathbb{Z}_2$ field inversion;
- reflections;
- $180^\circ$ rotations;
- periodic lattice translations.

The filename is currently spelled `Symmerty.py`. Rename it only if the corresponding import statements are also updated.

### `Time_embedding.py`

Implements Fourier-feature embeddings for conditioning the neural networks on stochastic time.

### `lr.py`

Implements a delayed cosine-decay learning-rate schedule. The learning rate remains fixed during an initial stage and then decays according to a cosine schedule.

### `train.py`

Main training script for minimizing the SPS path-space objective.

### `generation.py`

Restores a trained checkpoint and generates lattice field configurations using the learned forward stochastic process.

## Requirements

The code was developed with TensorFlow and TensorFlow Probability.

A representative environment is

```text
Python >= 3.10
TensorFlow == 2.15
TensorFlow Probability
NumPy
tqdm
```

Install the main dependencies with

```bash
pip install tensorflow==2.15 tensorflow-probability numpy tqdm
```

For GPU execution, install TensorFlow together with CUDA and cuDNN versions compatible with the local system.

## File preparation

The uploaded files should be renamed as follows before execution:

```text
Action(6).py              -> Action.py
CyclicConv2D(13).py       -> CyclicConv2D.py
DiffusionSubNet(67).py    -> DiffusionSubNet.py
generation(4).py          -> generation.py
lr(2).py                  -> lr.py
StochasticNet(15).py      -> StochasticNet.py
Symmerty(3).py            -> Symmerty.py
Time_embedding(7).py      -> Time_embedding.py
train(8).py               -> train.py
```

## Training

The main training parameters are configured at the end of `train.py`:

```python
L_list = [64]
Diffusion_list = [250]
Kappa = [0.28]
mb_size = 32
Ly = 8
```

The default quartic coupling is

```python
lam = 0.022
```

Run training with

```bash
python train.py
```

The training script creates checkpoints under

```text
model/phi4/kappa<KAPPA>_L<LX>_<NUM_DIFFUSION>/
```

For example,

```text
model/phi4/kappa0.28_L64_250/
```

The per-trajectory loss estimator is computed as

```python
kl_matrix = logP + action_val
```

and the batch objective is obtained from

```python
kl, std = tf.nn.moments(kl_matrix, -1)
```

where:

- `logP` is the trajectory-dependent endpoint log-density estimator;
- `action_val` is the terminal lattice action;
- `kl` is the batch mean of the SPS objective;
- `std` is the corresponding batch variance returned by `tf.nn.moments`.

The training log is written to

```text
training_log.txt
```

with the columns

```text
epoch    kl    std    time(s)
```

Checkpoints and generated diagnostic samples are saved every 250 epochs.

## Configuration generation

Before generation, set the parameters in `generation.py` consistently with the trained checkpoint:

```python
NUM_diffusion = 250
kappa = 0.28
lam = 0.022
mb_size = 4096
L = 64
```

Run

```bash
python generation.py
```

The script restores the latest checkpoint from

```text
model/phi4/kappa<KAPPA>_L<L>_<NUM_DIFFUSION>/
```

and saves generated arrays under

```text
kappa<KAPPA>L<L>/
├── DM.npy
├── logp.npy
└── action.npy
```

The files contain:

- `DM.npy`: generated terminal field configurations;
- `logp.npy`: trajectory-dependent log-density estimates;
- `action.npy`: terminal action values.

## Minimal generation example

```python
import tensorflow as tf

from Action import Action
from StochasticNet import StochasticNet

Lx = 64
Ly = 8
kappa = 0.28
lam = 0.022
num_diffusion = 250
num_samples = 4096

checkpoint_dir = "model/phi4/kappa0.28_L64_250"

model = StochasticNet(Lx=Lx, Ly=Ly)
checkpoint = tf.train.Checkpoint(model=model)
latest = tf.train.latest_checkpoint(checkpoint_dir)

if latest is None:
    raise FileNotFoundError(
        f"No TensorFlow checkpoint was found in {checkpoint_dir}"
    )

checkpoint.restore(latest).expect_partial()

action_fn = lambda field: Action(
    field,
    lam=lam,
    k=kappa,
)

configurations, log_probability, action_values = model.ForwardDiffusion(
    mb_size=num_samples,
    shape=(Lx, Ly),
    num_diffusion=num_diffusion,
    action_fn=action_fn,
)

print("Configurations:", configurations.shape)
print("Log probability:", log_probability.shape)
print("Action:", action_values.shape)
```

## Symmetry options

Random trajectory-level symmetry transformations can be enabled when the model is constructed:

```python
model = StochasticNet(
    Lx=Lx,
    Ly=Ly,
    use_symmetry=True,
)
```

The convolutional drift networks call the periodic convolution layers with

```python
use_d4=True
```

by default. For square kernels, the kernel is averaged over the $D_4$ group. For rectangular kernels, the code instead uses the shape-preserving $D_2$ subgroup.

## Numerical precision

The current implementation uses TensorFlow's default floating-point precision. Double precision can be enabled before model construction with

```python
tf.keras.backend.set_floatx("float64")
```

Double precision increases memory use and computational cost.

## Checkpoint consistency

The following settings should remain consistent between training and generation:

- lattice dimensions $L_x$ and $L_y$;
- hopping parameter $\kappa$;
- quartic coupling $\lambda$;
- diffusion depth $N$;
- hidden-channel dimensions;
- neural-network architecture;
- symmetry settings.

Changing the model architecture after training may lead to incomplete checkpoint restoration.

## Notes on the current implementation

1. The default numerical example uses a lattice of shape $L_x\times 8$.
2. The diffusion coefficient is generated by a learnable neural schedule.
3. The forward and backward drifts are represented by independent neural networks.
4. The generated endpoint density is trajectory dependent when exact trajectory balance is not achieved.
5. The supplied generation script uses

   ```python
   Step_generation = 20 * NUM_diffusion
   ```

   so the generation depth may differ from the depth used during training. Change this value deliberately according to the intended experiment.

## Citation

Please cite the associated paper when using this code:

```bibtex
@article{Chen:2026SPS,
  author        = {Chen, Shiyang and Qian, Moxian and Aarts, Gert and
                   Lucini, Biagio and Zhou, Kai},
  title         = {Stochastic Path Sampler for Lattice Field Theory},
  year          = {2026},
  eprint        = {2606.13790},
  archivePrefix = {arXiv},
  primaryClass  = {hep-lat},
  doi           = {10.48550/arXiv.2606.13790}
}
```



## Contact

For questions concerning the method, numerical experiments, or implementation, please contact the authors of the associated paper.
