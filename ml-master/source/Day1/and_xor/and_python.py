import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype="float32")
Y = np.array([[0.], [0.], [0.], [1.]], dtype="float32")

