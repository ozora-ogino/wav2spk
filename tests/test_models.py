import sys
sys.path.append('../')
import unittest
import numpy as np
import tensorflow as tf
from models.models import Wav2Spk


class TestWav2Spk(unittest.TestCase):
    def test_build(self):
        data = np.random.randn(100, 4000, 1)
        labels = np.random.randint(0, 2, 100)
        model = Wav2Spk()
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer="adam",
            metrics=["acc"],
        )
        model.fit(
            data, labels, epochs=3, batch_size=50, validation_split=0.2, verbose=0
        )


if __name__ == "__main__":
    unittest.main()
