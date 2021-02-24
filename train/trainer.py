import os
import sys

sys.path.append("../")

import time
from tensorflow.keras.optimizers import Adam

from utils import save_score, calculate_eer, record_eer


class Trainer(object):
    def __init__(self, model, name):
        self.model = model()
        self.time = time.strftime("(%Y-%d-%m %H:%M)")
        self.name = name

    def train(
        self,
        data,
        labels,
        batch_size=32,
        lr=0.001,
        loss="sparse_categorical_crossentropy",
        epochs=70,
        callbacks=None,
    ):

        self.model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=["acc"])

        history = self.model.fit(
            data,
            labels,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        self.save_history()

    def evaluate(self, data, labels):
        if not os.path.isdir("../scores/"):
            os.mkdir("../scores/")

        score_file = f"../scores/{self.name}-{self.time}.txt"
        pred = self.model.predict(data)
        score = pred[:, [0]] - pred[:, [1]]
        eer = calculate_eer(labels, score)
        save_score(score_file, score.flatten())
        return eer * 100

    def save_history(self):
        pass


class GeneratorTrainer(object):
    def __init__(self, model, directory, generator, name):
        self.model = model()
        self.train_dir = directory + "train/"
        self.val_dir = directory + "val/"
        self.generator = generator
        self.time = time.strftime("(%Y-%d-%m %H:%M)")
        self.name = name

    def train(
        self,
        batch_train=32,
        batch_val=32,
        target_len=30000,
        lr=0.001,
        loss="sparse_categorical_crossentropy",
        steps_per_epoch=30,
        epochs=20,
        validation_steps=40,
        callbacks=None,
    ):

        train_gen = self.generator()
        val_gen = self.generator()

        train_gen = train_gen.flow_from_directory(
            self.train_dir, batch_size=32, class_mode="binary", target_len=5000
        )

        val_gen = val_gen.flow_from_directory(
            self.train_dir, batch_size=32, class_mode="binary", target_len=5000
        )

        self.model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=["acc"])

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            steps_per_epochs=steps_per_epoch,
            epochs=epochs,
            validation_steps=validation_steps,
            verbose=1,
            callbacks=callbacks,
        )

        self.save_history()

    def evaluate(self, data, labels):
        if not os.path.isdir("../scores/"):
            os.mkdir("../scores")

        score_file = f"../scores/{self.name}-{self.time}.txt"
        pred = self.model.predict(data)
        score = pred[:, [0]] - pred[:, [1]]
        eer = calculate_eer(labels, score)
        save_score(score_file, score.flatten())
        return eer * 100

    def save_history(self):
        pass
