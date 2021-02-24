import sys

sys.path.append("../")

import os
import tensorflow as tf
from models.models import Wav2Spk
from utils import record_eer
from trainer import Trainer
from data_loader import get_test_data, get_dataset


# ================================================================

name = "wav2spk"
batch_size = 256
lr = 0.00001
epochs = 30 

path_ge = "../splited_train/irt-ge-1.bin"
path_sp = "../splited_train/irt-sp-1.bin"

# ================================================================




trainer = Trainer(Wav2Spk, name)

x_train, y_train = get_dataset(path_ge, path_sp)
trainer.train(
    x_train, y_train, batch_size=batch_size, lr=lr, epochs=epochs, callbacks=None
)

x_test, y_test = get_test_data()
eer = trainer.evaluate(data, labels)

print(f"EER: eer%")
