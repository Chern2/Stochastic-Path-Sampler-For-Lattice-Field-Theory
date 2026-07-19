# Stochastic Path Sampler for Lattice Field Theory

Official TensorFlow implementation of the **Stochastic Path Sampler (SPS)** for lattice field theory.

This repository contains the code used for the numerical experiments presented in:

> **Stochastic Path Sampler for Lattice Field Theory**
> Shiyang Chen, Moxian Qian, Gert Aarts, Biagio Lucini, and Kai Zhou
> arXiv:2606.13790 [hep-lat]

[Paper on arXiv](https://arxiv.org/abs/2606.13790)
[DOI: 10.48550/arXiv.2606.13790](https://doi.org/10.48550/arXiv.2606.13790)

## Overview

The Stochastic Path Sampler is a generative sampler for lattice field theory based on learnable nonequilibrium stochastic dynamics.

SPS constructs forward and auxiliary backward stochastic processes connecting a simple prior distribution to an unnormalized target distribution,

$\pi(\phi) = \frac{1}{Z}e^{-S(\phi)}$.

The forward and backward processes are trained by minimizing a path-space variational objective defined through the log-ratio of their trajectory measures. Once trained, the forward process generates independent proposals for the target lattice field distribution.

This implementation applies SPS to the two-dimensional scalar (\phi^4) theory with periodic boundary conditions.

## Lattice action

The lattice action implemented in this repository is


$S[\phi] =\sum_x \left[ (1-2\lambda)\phi_x^2 + \lambda\phi_x^4 - 2\kappa \sum_{\mu} \phi_x\phi_{x+\hat{\mu}} \right],$
]

where:

* (\kappa) is the hopping parameter;
* (\lambda) is the quartic coupling;
* periodic boundary conditions are imposed in every lattice direction.

The action is evaluated in `Action.py`.

## Stochastic dynamics

For a discretization depth (N), the forward transition is represented schematically by

[
\phi_{i+1}
==========

\phi_i
+
\sigma_\theta^2(t_i)
K_{\theta,\mathrm F}(\phi_i,t_i)
+
\sigma_\theta(t_i)\sqrt{\Delta t},\eta_i,
]

where

[
\eta_i\sim\mathcal N(0,I),
\qquad
\Delta t=\frac{1}{N}.
]

An auxiliary backward transition is defined using a second drift network,

[
\phi_i
======

\phi_{i+1}
+
\sigma_\theta^2(t_i)
K_{\theta,\mathrm B}(\phi_{i+1},t_i)
+
\sigma_\theta(t_i)\sqrt{\Delta t},\widetilde{\eta}_i.
]

The model accumulates the trajectory log-ratio

[
\log q_{\mathrm{SPS}}(\phi_N;\tau)
==================================

\log \pi_0(\phi_0)
+
\sum_{i=0}^{N-1}
\left[
\log q_{\mathrm F}
(\phi_{i+1}\mid\phi_i)
----------------------

\log q_{\mathrm B}
(\phi_i\mid\phi_{i+1})
\right].
]

The training objective is the reverse path-space KL estimator

[
\mathcal L_{\mathrm{SPS}}
=========================

\mathbb E_{q_{\mathrm F}}
\left[
\log q_{\mathrm{SPS}}(\phi_N;\tau)
+
S(\phi_N)
\right],
]

up to the unknown target normalization constant.

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

### `Action.py`

Implements the scalar (\phi^4) lattice action with periodic nearest-neighbour interactions.

### `CyclicConv2D.py`

Implements convolutional layers with periodic padding.

The layer supports:

* periodic boundary conditions;
* square and rectangular convolution kernels;
* optional (D_4) kernel symmetrization for square kernels;
* optional (D_2) kernel symmetrization for rectangular kernels;
* grouped convolution.

### `DiffusionSubNet.py`

Defines the neural drift network used for the forward and backward stochastic processes.

The architecture contains:

* periodic convolutional layers;
* residual connections;
* Fourier time embeddings;
* multiplicative time conditioning;
* batch normalization;
* bounded nonlinear activations;
* a learnable diffusion schedule.

### `StochasticNet.py`

Implements the complete stochastic path model, including:

* the prior Gaussian distribution;
* forward stochastic transitions;
* auxiliary backward transitions;
* trajectory log-probability accumulation;
* forward trajectory generation;
* evaluation of the terminal lattice action.

### `Symmerty.py`

Implements stochastic lattice symmetry transformations, including:

* global (\mathbb Z_2) field inversion;
* lattice reflections;
* (180^\circ) rotations;
* random periodic translations.

These transformations can be injected into the stochastic trajectory to improve symmetry coverage.

### `Time_embedding.py`

Implements Fourier-feature time embeddings for conditioning the drift and diffusion networks on the normalized stochastic time.

### `lr.py`

Implements a delayed cosine-decay learning-rate schedule. The learning rate remains constant during an initial training stage and subsequently follows cosine decay.

### `train.py`

Main training script for minimizing the SPS path-space objective.

### `generation.py`

Restores a trained checkpoint and generates lattice configurations using the learned forward stochastic process.

## Requirements

The implementation was developed using Python and TensorFlow.

A representative environment is:

```text
Python >= 3.10
TensorFlow == 2.15
TensorFlow Probability
NumPy
tqdm
```

Install the required packages with

```bash
pip install tensorflow==2.15 tensorflow-probability numpy tqdm
```

For GPU training, install a TensorFlow build compatible with the local CUDA and cuDNN versions.

## File names

The Python imports assume the following file names:

```text
Action.py
CyclicConv2D.py
DiffusionSubNet.py
StochasticNet.py
Symmerty.py
Time_embedding.py
lr.py
train.py
generation.py
```

Rename downloaded files accordingly before running the code.

In particular, retain the current spelling

```text
Symmerty.py
```

unless the corresponding imports are also changed to

```python
from Symmetry import *
```

## Training

The main parameters are configured near the end of `train.py`:

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

The code creates checkpoints under

```text
model/phi4/kappa<KAPPA>_L<LATTICE_SIZE>_<NUM_DIFFUSION>/
```

For example,

```text
model/phi4/kappa0.28_L64_250/
```

The training objective is evaluated as

```python
kl_matrix = logP + action_val
kl = tf.reduce_mean(kl_matrix)
```

where:

* `logP` is the path-dependent endpoint log-density estimator;
* `action_val` is (S(\phi_N));
* `kl` estimates the path-space variational objective.

The training log is saved as

```text
training_log.txt
```

with the columns

```text
epoch    kl    std    time(s)
```

## Configuration generation

After training, configure `generation.py` consistently with the checkpoint:

```python
NUM_diffusion = 250
kappa = 0.28
lam = 0.022
mb_size = 4096
L = 64
```

Generate configurations with

```bash
python generation.py
```

The generated arrays are saved as

```text
kappa<KAPPA>L<LATTICE_SIZE>/
├── SPS.npy
├── logp.npy
└── action.npy
```

Their contents are:

* `SPS.npy`: generated terminal field configurations;
* `logp.npy`: path-dependent log-density estimates;
* `action.npy`: terminal action values.

## Example

A minimal example for generating configurations from a trained model is

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

checkpoint_path = "model/phi4/kappa0.28_L64_250"

model = StochasticNet(Lx=Lx, Ly=Ly)

checkpoint = tf.train.Checkpoint(model=model)
latest = tf.train.latest_checkpoint(checkpoint_path)

if latest is None:
    raise FileNotFoundError(
        f"No TensorFlow checkpoint was found in {checkpoint_path}"
    )

checkpoint.restore(latest).expect_partial()

action_fn = lambda field: Action(
    field,
    lam=lam,
    k=kappa,
)

configurations, log_probability, action_values = (
    model.ForwardDiffusion(
        mb_size=num_samples,
        shape=(Lx, Ly),
        num_diffusion=num_diffusion,
        action_fn=action_fn,
    )
)

print("Configurations:", configurations.shape)
print("Log probability:", log_probability.shape)
print("Action:", action_values.shape)
```

## Symmetry options

Trajectory-level random symmetry transformations can be enabled when constructing the model:

```python
model = StochasticNet(
    Lx=Lx,
    Ly=Ly,
    use_symmetry=True,
)
```

The convolutional drift networks also use symmetry-averaged kernels by default through

```python
use_d4=True
```

For rectangular kernels, the convolution layer automatically uses the shape-preserving (D_2) subgroup instead of the complete (D_4) group.

## Numerical precision

The current implementation uses the default TensorFlow floating-point precision. For lattice field theory calculations requiring improved numerical accuracy, the global precision may be changed before model construction:

```python
tf.keras.backend.set_floatx("float64")
```

Using double precision increases memory consumption and computational cost.

## Checkpoint consistency

The following quantities must remain consistent between training and generation:

* lattice dimensions (L_x) and (L_y);
* hopping parameter (\kappa);
* quartic coupling (\lambda);
* diffusion depth (N);
* neural-network architecture;
* symmetry settings.

Changing any architecture parameter after training may prevent checkpoint restoration or produce partially restored models.

## Citation

Please cite the following paper when using this code:

```bibtex
@article{chen2026stochastic,
  title         = {Stochastic Path Sampler for Lattice Field Theory},
  author        = {Chen, Shiyang and Qian, Moxian and Aarts, Gert and
                   Lucini, Biagio and Zhou, Kai},
  year          = {2026},
  eprint        = {2606.13790},
  archivePrefix = {arXiv},
  primaryClass  = {hep-lat},
  doi           = {10.48550/arXiv.2606.13790}
}
```

## License

No license has currently been specified.

Before redistributing or modifying the code, add an appropriate open-source license according to the authors' intended terms of use.

## Contact

For questions about the method or implementation, please contact the authors of the associated paper.
