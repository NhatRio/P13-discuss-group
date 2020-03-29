import os
import logging

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from utils import data_preparation, build_fc_nn_model, compute_l1_loss, plot_data

logging.basicConfig(format='%(asctime)s [%(levelname)s] - [%(name)s] : %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

NUM_TRAIN = 50000
NUM_VALIDATION = 5000
POLY_COEFS = [0.5, 2, -1.25, 0.75]  # y = 0.5 + 2*x - 1.25*x^2 +0.75*x^3 = 0.5*1 + 2*x1 -1.25*x2 +0.75*x3
LOG_DIR = os.path.join(os.path.dirname(__file__), 'tflogdir')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

"""
-training data: 50000 samples
    ((1, x1, x2, x3) , y)
- linear regression: y = a0 + a1x1 + a2x2+ a3x3 =[1, x1, x2, x3] * [a0, a1, a2, a3]^T 
    + find [a0, a1, a2, a3] based on training data (x_train, y_train)
    + use them to predict a new input sample, e.g. x_val
- NN
    + Build a architecture of NN.
    + 
"""

def generate_data():
    x_train, y_train = data_preparation(num_data=NUM_TRAIN, poly_coefs=POLY_COEFS)
    x_val, y_val = data_preparation(num_data=NUM_VALIDATION, poly_coefs=POLY_COEFS)
    return {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}


def train_with_linearRegression_sklearn(data):
    linearRegres = LinearRegression()
    # fitting
    linearRegres.fit(data["x_train"], data["y_train"])
    # predict
    y_pred = linearRegres.predict(data["x_val"])
    return y_pred


def train_with_tensorflow(data, summary=False):
    # build model
    model = build_fc_nn_model(input_shape=(data["x_train"].shape[1],), output_features=1, summary=summary)

    # compile model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.MeanAbsoluteError())  # L1 loss

    # callback options.
    callbacks = [
        # Write TensorBoard logs to log directory. It works on a machine with an external Nvidia GPU.
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    ]

    # training
    logger.info('Fit model on training data')
    history = model.fit(data["x_train"], data["y_train"],
                        batch_size=256,
                        epochs=5,
                        ## Uncomment the following line if you want to run validation after each epoch
                        # validation_data=(data["x_val"], data["y_val"]),
                        ## Active the following line if you have a Nvidia GPU (and already install all drivers).
                        # callbacks=callbacks,
                        )
    if summary:
        logger.info('\nhistory dict:', history.history)
    return model.predict(data["x_val"], batch_size=10)



if __name__ == "__main__":
    logger.info("Generate data")
    data = generate_data()

    logger.info("Simple solution: Linear regression")
    y_pred_lr = train_with_linearRegression_sklearn(data)
    lr_loss = compute_l1_loss(data["y_val"], y_pred_lr)

    logger.info("Using neural network")
    y_pred_nn = train_with_tensorflow(data, summary=False)
    nn_loss = compute_l1_loss(data["y_val"], np.squeeze(y_pred_nn))

    logger.info("Comparison of average L1-loss")
    logger.info(f"Using Linear Regression model: {lr_loss}")
    logger.info(f"Using neural network: {nn_loss}")

    # Illustration for simple case (2 dimensions)
    if len(POLY_COEFS) == 2:
        specs = {
            f"linearRegr_{lr_loss}": (y_pred_lr, "r+"),
            f"neuralNetwork_{nn_loss}": (y_pred_nn, "b*")
        }
        plot_data(data["x_val"][:,1], specs)

