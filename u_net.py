# =================================================================================================
# U-Net 1D Architecture & Noise Generator Module
#
# Developed for paper: Robust U-Net Reconstruction of NMR Signals and Spectra
# in Noisy and Inhomogeneous Magnetic Fields

# Authors: Alfredo Aguilar, Lautaro Piermartini, and Esteban Anoardo
# Laboratorio de Relaxometría y Técnicas Especiales (LaRTE) - UNC-FaMAF - 2026.
# =================================================================================================
#______________________________________________________________________________________
import numpy as np
from typing import Optional
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import tensorflow as tf
#______________________________________________________________________________________

def build_unet_1d(input_length, n_channels, base_filters, depth, kernel_size, dropout):
    inputs = layers.Input(shape=(input_length, n_channels))
    x = inputs
    skips = []

    ####################################################################
    # calculate necessary padding
    mult = 2 ** depth
    pad_len = (-input_length) % mult
    if pad_len > 0:
        left_pad = pad_len // 2
        right_pad = pad_len - left_pad
    else:
        left_pad = right_pad = 0

    # effective input length for the encoder (input + pad_len if any)
    effective_input = input_length + pad_len

    # apply reflect padding if needed (symmetric)
    if pad_len > 0:
        x = layers.Lambda(lambda z: tf.pad(z, [[0, 0], [left_pad, right_pad], [0, 0]], mode='REFLECT')  )(x)
    ####################################################################


    ####################################################################
    ####################################################################
    # encoder
    encoder_blocks = []
    for d in range(depth):

        filters = base_filters * (2 ** d)

        block = Sequential([
            layers.Conv1D(filters, kernel_size, padding='same', activation='swish'),

            layers.Conv1D(filters, kernel_size, padding='same', activation='swish')],

            name=f'encoder_block_{d}')

        encoder_blocks.append(block)

    for d, block in enumerate(encoder_blocks):
        x = block(x)
        skips.append(x)
        x = layers.MaxPooling1D(2)(x)
       #x = layers.Dropout(dropout)(x)  # does not improve results!


    ####################################################################
    ####################################################################
    # bottleneck
    filters = base_filters * (2 ** depth)
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='swish')(x)
    #x = layers.Dropout(dropout)(x)  # does not improve results!
    ####################################################################
    ####################################################################


    ####################################################################
    ####################################################################
    # decoder: reconstruction with upsampling and skip concatenation
    for d in reversed(range(depth)):
        filters = base_filters * (2 ** d)
        x = layers.UpSampling1D(2)(x)

        # conv after upsampling
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='swish')(x)

        # retrieve corresponding skip connection
        skip = skips[d]

        x = layers.Concatenate(axis=-1)([x, skip])

        # post-concat convs (same as encoder)

        x = layers.Conv1D(filters, kernel_size, padding='same', activation='swish')(x)

        x = layers.Conv1D(filters, kernel_size, padding='same', activation='swish')(x)
    ####################################################################
    ####################################################################

    # crop the reflect padding applied at the beginning (if any)
    if pad_len > 0:
        # remove  left_pad and right_pad
        x = layers.Cropping1D((left_pad, right_pad))(x)


    ####################################################################
    outputs = layers.Conv1D(n_channels, kernel_size=1, activation='linear')(x)  # output: reconstruct intensity channel


    model = models.Model(inputs, outputs)


    return model


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def add_gaussian_noise(X: np.ndarray, sigma: float, random_state: Optional[int] = None) -> np.ndarray:
    # adds additive gaussian noise N(0, sigma^2) to X.                  X: (N, p)

    rng = np.random.default_rng(random_state)

    return X + rng.normal(0.0, sigma, size=X.shape).astype(np.float32)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
