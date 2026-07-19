import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class CyclicPadding(Layer):
    """
    Periodic padding for NHWC tensors.

    Parameters
    ----------
    padding : int, tuple/list of length 2, or tuple/list of length 4
        int:
            same padding on all four sides.
        (pad_h, pad_w):
            symmetric padding in height and width.
        (top, bottom, left, right):
            explicit padding on each side.
    """

    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)

        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            self.padding = (
                int(padding[0]),
                int(padding[0]),
                int(padding[1]),
                int(padding[1]),
            )

        elif isinstance(padding, (tuple, list)) and len(padding) == 4:
            self.padding = tuple(int(v) for v in padding)

        else:
            raise ValueError(
                "padding must be an int, a length-2 tuple/list, "
                "or a length-4 tuple/list"
            )

        if any(v < 0 for v in self.padding):
            raise ValueError(f"padding values must be non-negative: {self.padding}")

    def call(self, inputs):
        top, bottom, left, right = self.padding
        x = inputs

        if top > 0 or bottom > 0:
            parts = []
            if top > 0:
                parts.append(x[:, -top:, :, :])
            parts.append(x)
            if bottom > 0:
                parts.append(x[:, :bottom, :, :])
            x = tf.concat(parts, axis=1)

        if left > 0 or right > 0:
            parts = []
            if left > 0:
                parts.append(x[:, :, -left:, :])
            parts.append(x)
            if right > 0:
                parts.append(x[:, :, :right, :])
            x = tf.concat(parts, axis=2)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


def d2_transforms(kernel):
    """
    Generate the four shape-preserving D2 transforms.

    Valid for both square and rectangular kernels.

    kernel shape:
        [kh, kw, in_channels, out_channels]

    D2 elements:
        identity
        flip height
        flip width
        flip height + width, i.e. 180-degree rotation
    """
    k0 = kernel
    k_flip_h = tf.reverse(kernel, axis=[0])
    k_flip_w = tf.reverse(kernel, axis=[1])
    k_rot180 = tf.reverse(kernel, axis=[0, 1])

    return [
        k0,
        k_flip_h,
        k_flip_w,
        k_rot180,
    ]


def d4_transforms(kernel):
    """
    Generate all eight D4 transforms of a square convolution kernel.

    kernel shape:
        [kh, kw, in_channels, out_channels]

    Requirement:
        kh == kw
    """
    k0 = kernel

    k_t = tf.transpose(kernel, perm=[1, 0, 2, 3])

    # rotations
    k_rot90 = tf.reverse(k_t, axis=[0])
    k_rot180 = tf.reverse(kernel, axis=[0, 1])
    k_rot270 = tf.reverse(k_t, axis=[1])

    # reflections
    k_flip_h = tf.reverse(kernel, axis=[0])
    k_flip_w = tf.reverse(kernel, axis=[1])
    k_diag = k_t
    k_anti_diag = tf.reverse(k_t, axis=[0, 1])

    return [
        k0,
        k_rot90,
        k_rot180,
        k_rot270,
        k_flip_h,
        k_flip_w,
        k_diag,
        k_anti_diag,
    ]


class CyclicConv2D(Layer):
    """
    2D convolution with periodic boundary conditions.

    Features
    --------
    1. Periodic padding.
    2. Rectangular or square kernels.
    3. Optional D4/D2 symmetrisation:
        - if use_d4=True and kh == kw: use D4
        - if use_d4=True and kh != kw: use D2
        - if use_d4=False: no symmetrisation
    4. Optional weight normalization:
        W_eff = g * W / ||W||
    5. Optional grouped convolution.

    Input format
    ------------
    NHWC: [batch, height, width, channels]
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        use_bias=False,
        activation=None,
        groups=1,
        dilation_rate=1,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = int(filters)
        self.kernel_size = self._to_pair(kernel_size, "kernel_size")
        self.strides = self._to_pair(strides, "strides")
        self.dilation_rate = self._to_pair(dilation_rate, "dilation_rate")
        self.use_bias = bool(use_bias)
        self.activation = tf.keras.activations.get(activation)
        self.groups = int(groups)
        self.debug = bool(debug)

        if self.filters <= 0:
            raise ValueError(f"filters must be positive, got {self.filters}")

        if self.groups <= 0:
            raise ValueError(f"groups must be positive, got {self.groups}")

        if any(v <= 0 for v in self.kernel_size):
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")

        if any(v <= 0 for v in self.strides):
            raise ValueError(f"strides must be positive, got {self.strides}")

        if any(v <= 0 for v in self.dilation_rate):
            raise ValueError(
                f"dilation_rate must be positive, got {self.dilation_rate}"
            )

        kh, kw = self.kernel_size
        dh, dw = self.dilation_rate

        effective_kh = (kh - 1) * dh + 1
        effective_kw = (kw - 1) * dw + 1

        # Periodic analogue of SAME padding.
        pad_top = (effective_kh - 1) // 2
        pad_bottom = effective_kh - 1 - pad_top
        pad_left = (effective_kw - 1) // 2
        pad_right = effective_kw - 1 - pad_left

        self.pad = CyclicPadding(
            padding=(pad_top, pad_bottom, pad_left, pad_right),
            name="cyclic_padding",
        )

        self.kernel_base = None
        self.bias = None
        self.weight_g = None

    @staticmethod
    def _to_pair(value, name):
        if isinstance(value, int):
            return (int(value), int(value))

        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (int(value[0]), int(value[1]))

        raise ValueError(
            f"{name} must be an int or a tuple/list of length 2; got {value}"
        )

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "CyclicConv2D expects an NHWC rank-4 input; "
                f"received input_shape={input_shape}"
            )

        in_channels = input_shape[-1]

        if in_channels is None:
            raise ValueError("The input channel dimension must be known")

        in_channels = int(in_channels)

        if in_channels % self.groups != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible "
                f"by groups={self.groups}"
            )

        if self.filters % self.groups != 0:
            raise ValueError(
                f"filters={self.filters} must be divisible "
                f"by groups={self.groups}"
            )

        kh, kw = self.kernel_size

        kernel_shape = (
            kh,
            kw,
            in_channels // self.groups,
            self.filters,
        )

        self.kernel_base = self.add_weight(
            name="kernel_base",
            shape=kernel_shape,
            initializer=initializers.GlorotUniform(),
            trainable=True,
        )


        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )

        if self.debug:
            print(
                f"[CyclicConv2D.build] name={self.name}, "
                f"input_shape={input_shape}, "
                f"kernel_size={self.kernel_size}, "
                f"kernel_shape={kernel_shape}, "
                f"padding={self.pad.padding}, "
                f"weight_norm={self.weight_norm}"
            )

        super().build(input_shape)

    def _symmetrize_kernel(self, kernel, use_d4):
        """
        Apply D4/D2 symmetrisation if requested.

        If use_d4=True:
            square kernel      -> D4 average
            rectangular kernel -> D2 average

        If use_d4=False:
            return kernel unchanged
        """
        if not use_d4:
            return kernel, "None"

        kh, kw = self.kernel_size

        if kh == kw:
            transforms = d4_transforms(kernel)
            symmetry_name = "D4"
        else:
            transforms = d2_transforms(kernel)
            symmetry_name = "D2"

        kernel_eff = tf.add_n(transforms) / tf.cast(
            len(transforms),
            kernel.dtype,
        )

        return kernel_eff, symmetry_name


    def call(self, inputs, use_d4=False):
        x = self.pad(inputs)

        # 1. Start from raw trainable kernel.
        kernel_eff = self.kernel_base

        # 2. D4/D2 average.
        kernel_eff, symmetry_name = self._symmetrize_kernel(
            kernel_eff,
            use_d4=use_d4,
        )


        if self.debug:
            tf.print(
                "[CyclicConv2D.call]",
                self.name,
                "kernel_size=", self.kernel_size,
                "symmetry=", symmetry_name,
                "weight_norm=", self.weight_norm,
            )

        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilation_rate

        strides_tf = [1, stride_h, stride_w, 1]
        dilations_tf = [1, dilation_h, dilation_w, 1]

        if self.groups == 1:
            y = tf.nn.conv2d(
                x,
                kernel_eff,
                strides=strides_tf,
                padding="VALID",
                dilations=dilations_tf,
                data_format="NHWC",
            )

        else:
            input_groups = tf.split(x, self.groups, axis=-1)
            kernel_groups = tf.split(kernel_eff, self.groups, axis=-1)

            outputs = []

            for x_group, kernel_group in zip(input_groups, kernel_groups):
                y_group = tf.nn.conv2d(
                    x_group,
                    kernel_group,
                    strides=strides_tf,
                    padding="VALID",
                    dilations=dilations_tf,
                    data_format="NHWC",
                )
                outputs.append(y_group)

            y = tf.concat(outputs, axis=-1)

        if self.use_bias:
            y = tf.nn.bias_add(y, self.bias, data_format="NHWC")

        if self.activation is not None:
            y = self.activation(y)

        return y

    def compute_output_shape(self, input_shape):
        batch, height, width, _ = input_shape
        stride_h, stride_w = self.strides

        out_height = None if height is None else (height + stride_h - 1) // stride_h
        out_width = None if width is None else (width + stride_w - 1) // stride_w

        return (batch, out_height, out_width, self.filters)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "activation": tf.keras.activations.serialize(self.activation),
                "groups": self.groups,
                "dilation_rate": self.dilation_rate,
                "debug": self.debug,
                "weight_norm": self.weight_norm,
            }
        )
        return config
