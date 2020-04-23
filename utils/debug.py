import keras.layers as layers
import tensorflow as tf


def print_layer(layer, message, first_n=80, summarize=10000000):
    return layers.Lambda((
        lambda x: tf.compat.v1.Print(x, [x],
                                     message=message,
                                     first_n=first_n,
                                     summarize=summarize)))(layer)