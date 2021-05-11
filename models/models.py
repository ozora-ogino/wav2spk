import tensorflow_addons as tfa
import tensorflow as tf


class TemporalGating(tf.keras.layers.Layer):
    def __init__(self):
        super(TemporalGating, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense_1 = None
        self.dense_2 = None


def call(self, inputs):
    x = self.pooling(inputs)
    x = tf.keras.layers.Reshape((1, inputs.shape[-1]))(x)
    if not self.dense_1:
        self.dense_1 = tf.keras.layers.Dense(
            inputs.shape[-1] // 16,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False,
        )
    x = self.dense_1(x)
    if not self.dense_2:
        self.dense_2 = tf.keras.layers.Dense(
            inputs.shape[-1],
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )
    x = self.dense_2(x)
    x = tf.keras.layers.Multiply()([inputs, x])
    return x


class Wav2Spk(tf.keras.models.Model):
    def __init__(self):
        super(Wav2Spk, self).__init__()
        # Encoder
        self.conv_1 = tf.keras.layers.Conv1D(
            40, kernel_size=10, strides=5, padding="same"
        )
        self.norm_1 = tfa.layers.InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
        self.relu_1 = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv1D(
            200, kernel_size=5, strides=4, padding="same"
        )
        self.norm_2 = tfa.layers.InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
        self.relu_2 = tf.keras.layers.ReLU()
        self.conv_3 = tf.keras.layers.Conv1D(
            300, kernel_size=5, strides=2, padding="same"
        )
        self.norm_3 = tfa.layers.InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
        self.relu_3 = tf.keras.layers.ReLU()
        self.conv_4 = tf.keras.layers.Conv1D(
            512, kernel_size=3, strides=2, padding="same"
        )
        self.norm_4 = tfa.layers.InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
        self.relu_4 = tf.keras.layers.ReLU()
        self.conv_5 = tf.keras.layers.Conv1D(
            512, kernel_size=3, strides=2, padding="same"
        )
        self.norm_5 = tfa.layers.InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )

        # Frames Aggregator
        self.conv_6 = tf.keras.layers.Conv1D(
            512, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu
        )
        self.conv_7 = tf.keras.layers.Conv1D(
            512, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu
        )
        self.conv_8 = tf.keras.layers.Conv1D(
            512, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu
        )
        self.conv_9 = tf.keras.layers.Conv1D(
            512, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu
        )

        self.gating = TemporalGating()

        # This will be chagned to Statics pooling layer in a future.
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        # Utt.Layers
        self.dense_10 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense_11 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense_12 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.norm_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.norm_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.norm_4(x)
        x = self.relu_4(x)

        x = self.conv_5(x)
        x = self.norm_5(x)

        x = self.gating(x)

        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)

        x = self.pooling(x)

        x = self.dense_10(x)
        x = self.dense_11(x)

        return self.dense_12(x)