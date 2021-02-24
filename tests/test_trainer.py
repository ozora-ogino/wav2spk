import os
import sys

sys.path.append("../")
import unittest
import numpy as np

from models.models import Wav2Spk
from train.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def test_train(self):
        trainer = Trainer(Wav2Spk, "wav2spk")
        x_train = np.random.randn(100, 1000, 1)
        y_train = np.random.randint(0, 2, 100)
        trainer.train(x_train, y_train, epochs=3)

    def test_eval(self):
        trainer = Trainer(Wav2Spk, "wav2spk")
        x_test = np.random.randn(100, 1000, 1)
        y_test = np.random.randint(0, 2, 100)
        trainer.evaluate(x_test, y_test)
        os.remove(f"../scores/{trainer.name}-{trainer.time}.txt")


if __name__ == "__main__":
    unittest.main()
