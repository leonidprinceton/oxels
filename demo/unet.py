import tensorflow as tf
from tensorflow import keras

from keras import layers


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")

def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(x):
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=2,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply

def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x

    return apply


def build_unet(
    height,
    width,
    depth=5,
    final_channels=32,
    num_res_blocks=3,
    norm_groups=8,
    first_conv_channels=8,
    activation_fn=keras.activations.swish,
):
    input = layers.Input(shape=(height, width, 1), name="input")
    
    x = input
    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(2, 2),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)
    skips = [x]
    
    #This is the downsampling stream of the UNet
    for i in range(depth):
        for _ in range(num_res_blocks):
            x = ResidualBlock(first_conv_channels*2**i, groups=norm_groups, activation_fn=activation_fn)(x)
            skips.append(x)

        if i != depth-1:
            x = DownSample(first_conv_channels*2**(i+1))(x)
            skips.append(x)

    #This is the upsampling stream of the UNet
    for i in reversed(range(depth)):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(first_conv_channels*2**depth, groups=norm_groups, activation_fn=activation_fn)(x)

        if i != 0:
            x = UpSample(first_conv_channels*2**depth)(x)

    x = layers.Conv2D(final_channels, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(input, x)     
    return model
