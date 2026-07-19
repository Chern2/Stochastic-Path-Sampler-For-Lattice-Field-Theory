import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def symmetry(inputs, prob=0.5):
    """
    Apply random lattice symmetries to inputs for data augmentation.

    This function applies three types of symmetries:
    1. Global Z₂ sign flip (spin inversion)
    2. D₄ dihedral group transformations (rotations and reflections)
    3. Random cyclic translations (toroidal boundary conditions)

    Parameters
    ----------
    inputs : tf.Tensor, shape [batch, Lx, Ly, channels]
            Input lattice configurations

    Returns
    -------
    tf.Tensor, shape [batch, Lx, Ly, channels]
            Transformed inputs with random symmetries applied
    """

    mb_size = inputs.shape[0]
    Lx = inputs.shape[1]
    Ly = inputs.shape[2]

    # 1. Random sign flip (global Z₂ symmetry)
    # Randomly flip the sign of all spins with 50% probability
    rand_sign = tf.random.uniform(shape=(mb_size, 1, 1, 1))
    inputs = tf.where(rand_sign > prob, inputs, -inputs)
    #flip = tf.cast(tfp.distributions.Bernoulli(probs=0.5).sample([tf.shape(inputs)[0], 1, 1, 1]), inputs.dtype)
    #inputs = (1.0 - 2.0*flip) * inputs

    #if Lx == Ly:
    # Create all 8 D4 transformations manually
    # D₄ group includes: identity, 90°, 180°, 270° rotations,
    # horizontal/vertical flips, and diagonal reflections
    identity = inputs  # No transformation
    #rot90 = tf.transpose(tf.reverse(inputs, axis=[2]), [0, 2, 1, 3])  # 90° rotation
    rot180 = tf.reverse(tf.reverse(inputs, axis=[1]), axis=[2])  # 180° rotation
    #rot270 = tf.reverse(tf.transpose(inputs, [0, 2, 1, 3]), axis=[2])  # 270° rotation
    flip_h = tf.reverse(inputs, axis=[2])  # Horizontal reflection
    flip_v = tf.reverse(inputs, axis=[1])  # Vertical reflection
    # Main diagonal reflection: flip_h then rot90
    flip_diag1 = tf.transpose(tf.reverse(flip_h, axis=[2]), [0, 2, 1, 3])
    # Anti-diagonal reflection: flip_h then rot270
    flip_diag2 = tf.reverse(tf.transpose(flip_h, [0, 2, 1, 3]), axis=[2])

    # Stack all D₄ transformations along a new dimension
    d4_transforms = tf.stack([
        identity, 
        #rot90, 
        rot180, 
        #rot270,
        flip_h, flip_v, #flip_diag1, flip_diag2
    ], axis=1)
    # Randomly select one D₄ transformation per batch element
    indices = tf.stack([
        tf.range(mb_size),
        tf.random.uniform([mb_size], 0, 4, dtype=tf.int32)
    ], axis=1)

    inputs = tf.gather_nd(d4_transforms, indices)


#    else:
#        inputs = tf.reverse(inputs, axis=[2])  # Horizontal reflection
#        inputs = tf.reverse(inputs, axis=[1])  # Vertical reflection

    # 3. Random translation (cyclic shift) - periodic boundary conditions
    shift_x = tf.random.uniform([mb_size], 0, Lx, dtype=tf.int32)
    shift_y = tf.random.uniform([mb_size], 0, Ly, dtype=tf.int32)



    xx = tf.range(Lx)[None, :, None]  # [1, Lx, 1]
    yy = tf.range(Ly)[None, None, :]  # [1, 1, Ly]

    xx = tf.tile(xx, [mb_size, 1, Ly])      # [batch, Lx, Ly]
    yy = tf.tile(yy, [mb_size, Lx, 1])      # [batch, Lx, Ly]

    # Apply cyclic shifts with modulo operation
    xx = (xx - shift_x[:, None, None]) % Lx
    yy = (yy - shift_y[:, None, None]) % Ly

    indices = tf.stack([xx, yy], axis=-1)

    # Create batch indices and combine with spatial indices
    batch_idx = tf.range(mb_size)[:, None, None, None]
    batch_idx = tf.tile(batch_idx, [1, Lx, Ly, 1])
    gather_idx = tf.concat([batch_idx, indices], axis=-1)

    # Apply the translation using gather_nd
    inputs = tf.gather_nd(inputs, gather_idx)

    return inputs

def Z2_symmetry(inputs):
    mb_size = tf.shape(inputs)[0]
    Lx = tf.shape(inputs)[1]
    Ly = tf.shape(inputs)[2]

    # 1. Random sign flip (global Z₂ symmetry)
    # Randomly flip the sign of all spins with 50% probability
    rand_sign = tf.random.uniform(shape=(mb_size, 1, 1, 1))
    inputs = tf.where(rand_sign > 0.5, inputs, -inputs)
    return inputs
