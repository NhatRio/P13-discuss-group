import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_core.python.keras import layers

def data_preparation(num_data, poly_coefs, x_range=1):
    """
    num_data: number of samples will be generated
    poly_coefs: 1D array contains coefficients of a polynomial model.
    x_range: the range in which samples will be drawn.
        - randomly drawn 1D array x in range [-x_range; x_range]
        - y = poly_coefs[0] + poly_coefs[1] * x + poly_coefs[2] * x * x + ...
       """
    # Sanity check
    poly_coefs = np.array(poly_coefs, dtype=np.float32)
    assert poly_coefs.ndim == 1 and len(poly_coefs) > 1, "Poly_coefs must be a 1D array with at least 2 elements."
    # Randomly draw in range [-10;10]
    x_rand = 2 * x_range * np.random.rand(num_data, 1) - x_range
    # Create matrix [1, x, x^2, ...]
    data_x = np.tile(x_rand, reps=(1, len(poly_coefs)))
    data_x = np.power(data_x, np.arange(len(poly_coefs)))
    # compute y
    data_y = np.matmul(data_x, poly_coefs)
    # test to make sure the code is good
    np.testing.assert_array_almost_equal(data_y[..., np.newaxis], np.polynomial.polynomial.polyval(x_rand, poly_coefs))
    # add noise N(0, 0.25^2)
    data_y = data_y + 0.25 * np.random.randn(*data_y.shape)
    return data_x, data_y

def build_fc_nn_model(input_shape, output_features, summary=False):
    """Build fully connected neural network model"""
    inputs = tf.keras.Input(input_shape, name="input")
    # todo: use leakyReLu to preserve negative values
    hid_layer1 = layers.Dense(units=16, activation="relu", name="dense1")(inputs)
    outputs = layers.Dense(units=output_features, name="output")(hid_layer1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if summary:
        model.summary()
    return model

def compute_l1_loss(y, y_pred):
    return tf.reduce_mean(tf.abs(y - y_pred))

def plot_data(x, y_preds):
    plt.figure()
    for label, specs in y_preds.items():
        plt.plot(x, specs[0], specs[1], label=label)
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.legend()
    plt.show()
